"""Open question: does a LEARNED expert graph beat a CONSTRUCTED one?

SymphonySMoE (arXiv 2510.16411) builds its expert graph from weight similarity,
thresholded and static. This DCN learns node positions by gradient descent and
reads the graph off them. Those are different designs, and SymphonySMoE does not
compare against a learned adjacency -- so this comparison is unanswered in the
literature.

CONDITIONS (identical data / init / epochs; recruitment ON throughout, since the
graph only matters when it is used to recruit):

  learned   cosine sim of `positions`, trained by gradient descent    (ours)
  weights   cosine sim of node first-layer weights, detached          (SymphonySMoE-style)
  frozen    cosine sim of `positions` pinned at random init           (control)

WHY 'frozen' MATTERS: it separates two claims that are easy to conflate --
  (a) recruitment needs a graph at all, vs
  (b) that graph specifically needs to be LEARNED.
If frozen ~= learned, the graph merely needs to exist and gradient descent on
positions is wasted machinery. If learned > frozen > seeds-only, the learning
is doing real work.

Usage:
  python graph_mode_comparison.py                  # CPU smoke
  python graph_mode_comparison.py --full --seeds 3 # GPU, error bars
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.arithmetic import create_datasets, PAD_IDX, classify_expression
from src.models.coalition import CoalitionModel
from src.training.losses import coalition_load_balance_loss, coalition_size_loss

FULL = '--full' in sys.argv
N_SEED_RUNS = 1
for i, a in enumerate(sys.argv):
    if a == '--seeds':
        N_SEED_RUNS = int(sys.argv[i + 1])

if FULL:
    D_MODEL, N_LAYERS, D_NODE = 256, 4, 64
    TRAIN_N, EPOCHS, BS = 80000, 60, 128
else:
    D_MODEL, N_LAYERS, D_NODE = 160, 3, 48
    TRAIN_N, EPOCHS, BS = 25000, 50, 128

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODES = ['learned', 'weights', 'frozen']


def base_cfg():
    c = yaml.safe_load(open('configs/experiment.yaml'))
    c['data']['train_size'] = TRAIN_N
    c['data']['val_size'] = 2000
    c['data']['test_size'] = 5000
    c['data']['gen_test_size'] = 2000
    c['model']['d_model'] = D_MODEL
    c['model']['n_layers'] = N_LAYERS
    c['coalition']['d_node'] = D_NODE
    return c


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    c = t = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = model(x)['logits'].argmax(dim=-1)
        m = (p == y) | (y == PAD_IDX)
        c += m.all(dim=-1).sum().item(); t += y.size(0)
    return c / max(t, 1)


@torch.no_grad()
def graph_stats(model, loader):
    """Activation mass, plus how structured the graph actually is."""
    model.eval()
    mass = []
    for x, _ in loader:
        for aux in model(x.to(DEVICE)).get('aux_data', []):
            if aux['type'] == 'coalition':
                mass.append(aux['node_activation'].sum(dim=1))
        break
    sim = None
    for layer in model.layers:
        if hasattr(layer.ffn, 'positions'):
            s = model(next(iter(loader))[0][:2].to(DEVICE))['aux_data'][0]['similarity_matrix']
            n = s.shape[0]
            off = s[~torch.eye(n, dtype=torch.bool, device=s.device)]
            sim = (off.mean().item(), off.std().item(), (off > 0.7).sum().item())
            break
    return (torch.cat(mass).mean().item() if mass else 0.0), sim


@torch.no_grad()
def type_similarity(model, datasets, n=800):
    from collections import defaultdict
    model.eval()
    acts = defaultdict(list)
    for i in range(min(n, len(datasets['test']))):
        x, _ = datasets['test'][i]
        expr, _ = datasets['test'].data[i]
        out = model(x.unsqueeze(0).to(DEVICE))
        acts[classify_expression(expr)].append(
            out['aux_data'][0]['node_activation'][0].cpu().numpy())
    means = {t: np.mean(v, axis=0) for t, v in acts.items() if v}
    ts = list(means); worst = 0.0
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            a, b = means[ts[i]], means[ts[j]]
            worst = max(worst, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    return worst


def run(mode, seed, datasets, tl, testl, genl):
    torch.manual_seed(seed)
    cfg = base_cfg()
    model = CoalitionModel(cfg).to(DEVICE)
    model.set_graph_mode(mode)
    model.set_norm_mode('sum')      # magnitude-controlled, as in the ablation

    rp = model.get_routing_params(); rid = {id(p) for p in rp}
    nrp = [p for p in model.parameters() if id(p) not in rid]
    opt = torch.optim.AdamW([{'params': nrp, 'lr': 3e-4},
                             {'params': rp, 'lr': 3e-3}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=3e-5)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    cc = cfg['coalition']

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        model.set_temperature(model.compute_temperature(ep, EPOCHS))
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = crit(out['logits'].reshape(-1, out['logits'].size(-1)), y.reshape(-1))
            for aux in out.get('aux_data', []):
                if aux['type'] == 'coalition':
                    loss = loss + cc['balance_coef'] * coalition_load_balance_loss(aux['node_activation'])
                    loss = loss + cc['size_coef'] * coalition_size_loss(
                        aux['node_activation'], cc['target_coalition_size'])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.set_temperature(cfg['coalition']['temp_end'])

    acc, gen = evaluate(model, testl), evaluate(model, genl)
    mass, sim = graph_stats(model, testl)
    tsim = type_similarity(model, datasets)
    smean, sstd, sclose = sim if sim else (0, 0, 0)
    print(f'  seed{seed} {mode:8s} | acc {acc:.4f}  gen {gen:.4f}  mass {mass:.1f}  '
          f'graph(mean {smean:+.3f} std {sstd:.3f} close {sclose})  tsim {tsim:.3f}  '
          f'[{time.time()-t0:.0f}s]', flush=True)
    return dict(acc=acc, gen=gen, mass=mass, tsim=tsim, close=sclose)


def main():
    cfg = base_cfg()
    datasets = create_datasets(cfg)
    tl = DataLoader(datasets['train'], batch_size=BS, shuffle=True)
    testl = DataLoader(datasets['test'], batch_size=256)
    genl = DataLoader(datasets['gen_test'], batch_size=256)

    print('=== GRAPH MODE COMPARISON: learned vs constructed ===')
    print(f'device={DEVICE}  d={D_MODEL} L={N_LAYERS} {TRAIN_N} ex, {EPOCHS} ep, '
          f'{N_SEED_RUNS} seed(s)\n', flush=True)

    seeds = [1234 + 1000 * i for i in range(N_SEED_RUNS)]
    res = {m: [] for m in MODES}
    for s in seeds:
        for m in MODES:
            res[m].append(run(m, s, datasets, tl, testl, genl))
        print('', flush=True)

    def agg(m, k):
        v = [r[k] for r in res[m]]
        return np.mean(v), (np.std(v) if len(v) > 1 else 0.0)

    print(f'{"mode":10s} {"test acc":>16} {"gen acc":>16} {"type-sim":>9}')
    print('-' * 55)
    for m in MODES:
        am, asd = agg(m, 'acc'); gm, gsd = agg(m, 'gen'); tm, _ = agg(m, 'tsim')
        print(f'{m:10s} {am:>9.4f} ±{asd:.4f} {gm:>9.4f} ±{gsd:.4f} {tm:>9.3f}')

    L, W, Fz = (agg(m, 'acc')[0] for m in MODES)
    noise = agg('learned', 'acc')[1] + agg('weights', 'acc')[1]
    thresh = max(0.01, noise)

    print('\n=== VERDICT ===')
    print(f'  learned vs weights (SymphonySMoE-style): {L-W:+.4f}')
    print(f'  learned vs frozen  (is learning needed): {L-Fz:+.4f}')
    if len(seeds) > 1:
        print(f'  (combined seed noise ~±{noise:.4f})')

    if L - W > thresh:
        print('  >>> Learning the graph beats constructing it from weights.')
        print('      This is the gap SymphonySMoE leaves open.')
    elif W - L > thresh:
        print('  >>> Constructed graph WINS -- learned positions are not worth it.')
    else:
        print('  >>> No reliable difference: learned and constructed graphs tie.')

    if L - Fz <= thresh:
        print('  >>> NOTE: frozen random graph matches learned. The graph only needs')
        print('      to EXIST, not be learned -- gradient descent on positions is')
        print('      doing little. That would undercut the "self-organizing" claim.')


if __name__ == '__main__':
    main()
