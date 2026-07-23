"""v8: the ORIGINAL design — does a learned edge matrix make recruitment work?

v6/v7 used position-based cosine similarity for recruitment. That constrained
the graph to be spatial: nearby nodes got recruited, but "nearby" forced them
into redundancy (shared gradients → same function). The position approach was
a detour from the original architecture.

THE ORIGINAL IDEA:
  Each pair of nodes has a free learned edge weight (sigmoid of edge_logits).
  No positions, no proximity, no spatial constraint. The network learns WHICH
  nodes should co-fire by backpropagating through the edge weights directly.
  This can learn any structure: complementary pairs, chains, hubs, cliques.

CONDITIONS (identical data/init per seed):
  C_matchedk   recruit OFF, k=7   -- the bar to beat (same as v6/v7 control)
  B_seedsonly  recruit OFF, k=4   -- seeds alone, no recruitment
  v8_recruit   recruit ON,  k=4   -- original edge-matrix recruitment

SUCCESS:
  v8_recruit BEATS C_matchedk → recruitment with learned edges does something
  that plain top-k cannot. The edge matrix learned useful co-activation.

  v8_recruit ≈ C_matchedk → same result as v6/v7 — recruitment is still just
  an expensive way to raise k, regardless of the graph mechanism.

Usage:
  python edge_recruitment_test.py --seeds 3        # cheap config, ~3h CPU
  python edge_recruitment_test.py --full --seeds 3 # GPU
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
from collections import defaultdict
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

CONDITIONS = [
    ('C_matchedk',  False, 7),
    ('B_seedsonly', False, 4),
    ('v8_recruit',  True,  4),
]


def cfg_for(n_seeds):
    c = yaml.safe_load(open('configs/experiment.yaml'))
    c['data']['train_size'] = TRAIN_N
    c['data']['val_size'] = 2000
    c['data']['test_size'] = 5000
    c['data']['gen_test_size'] = 2000
    c['model']['d_model'] = D_MODEL
    c['model']['n_layers'] = N_LAYERS
    c['coalition']['d_node'] = D_NODE
    c['coalition']['n_seeds'] = n_seeds
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
def coalition_stats(model, loader):
    model.eval()
    mass, recr = [], []
    for x, _ in loader:
        for aux in model(x.to(DEVICE)).get('aux_data', []):
            if aux['type'] == 'coalition':
                mass.append(aux['node_activation'].sum(dim=1))
                recr.append(aux['recruit_activation'].sum(dim=1))
    return (torch.cat(mass).mean().item() if mass else 0.0,
            torch.cat(recr).mean().item() if recr else 0.0)


@torch.no_grad()
def type_similarity(model, datasets, n=800):
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


@torch.no_grad()
def edge_stats(model):
    """How structured is the learned graph? Report sparsity and variance."""
    for layer in model.layers:
        if hasattr(layer.ffn, 'edge_logits'):
            ew = torch.sigmoid(layer.ffn.edge_logits)
            mask = 1.0 - torch.eye(ew.shape[0], device=ew.device)
            vals = ew[mask.bool()]
            return {
                'mean': vals.mean().item(),
                'std': vals.std().item(),
                'strong': (vals > 0.8).sum().item(),
                'weak': (vals < 0.2).sum().item(),
            }
    return {}


def run(label, recruit, n_seeds, seed, datasets, tl, testl, genl):
    torch.manual_seed(seed)
    cfg = cfg_for(n_seeds)
    model = CoalitionModel(cfg).to(DEVICE)
    model.set_recruitment(recruit)
    model.set_norm_mode('sum')

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
    mass, recr = coalition_stats(model, testl)
    tsim = type_similarity(model, datasets)
    es = edge_stats(model) if recruit else {}

    edge_info = ''
    if es:
        edge_info = (f'  edges(mean={es["mean"]:.2f} std={es["std"]:.2f} '
                     f'strong={es["strong"]} weak={es["weak"]})')
    print(f'  seed{seed} {label:12s} k={n_seeds} recruit={str(recruit):5s} | '
          f'acc {acc:.4f}  gen {gen:.4f}  mass {mass:.1f}  recr {recr:.2f}  '
          f'tsim {tsim:.3f}  [{time.time()-t0:.0f}s]{edge_info}', flush=True)
    return dict(acc=acc, gen=gen, mass=mass, recr=recr, tsim=tsim, edges=es)


def main():
    base = cfg_for(4)
    datasets = create_datasets(base)
    tl = DataLoader(datasets['train'], batch_size=BS, shuffle=True)
    testl = DataLoader(datasets['test'], batch_size=256)
    genl = DataLoader(datasets['gen_test'], batch_size=256)

    print('=== v8: ORIGINAL EDGE-MATRIX RECRUITMENT TEST ===')
    print(f'device={DEVICE}  d={D_MODEL} L={N_LAYERS} {TRAIN_N} ex, {EPOCHS} ep, '
          f'{N_SEED_RUNS} seed(s)\n', flush=True)
    if N_SEED_RUNS < 3:
        print('NOTE: run with --seeds 3 for error bars.\n', flush=True)

    seeds = [1234 + 1000 * i for i in range(N_SEED_RUNS)]
    res = {c[0]: [] for c in CONDITIONS}

    for s in seeds:
        for label, recruit, k in CONDITIONS:
            res[label].append(run(label, recruit, k, s, datasets, tl, testl, genl))
        print('', flush=True)

    def agg(label, key):
        v = [r[key] for r in res[label]]
        return np.mean(v), (np.std(v) if len(v) > 1 else 0.0)

    print(f'\n{"condition":14s} {"test acc":>16} {"gen acc":>16} {"tsim":>9} {"mass":>7}')
    print('-' * 66)
    for label, _, _ in CONDITIONS:
        am, asd = agg(label, 'acc'); gm, gsd = agg(label, 'gen')
        tm, _ = agg(label, 'tsim'); mm, _ = agg(label, 'mass')
        print(f'{label:14s} {am:>9.4f} ±{asd:.4f} {gm:>9.4f} ±{gsd:.4f} {tm:>9.3f} {mm:>7.1f}')

    ctrl_acc = agg('C_matchedk', 'acc')[0]
    seed_acc = agg('B_seedsonly', 'acc')[0]
    v8_acc = agg('v8_recruit', 'acc')[0]
    v8_tsim = agg('v8_recruit', 'tsim')[0]
    ctrl_tsim = agg('C_matchedk', 'tsim')[0]
    noise = agg('C_matchedk', 'acc')[1] + agg('v8_recruit', 'acc')[1]
    thresh = max(0.01, noise)

    print(f'\n=== VERDICT ===')
    print(f'  v8 (edge recruit) vs seeds-only:    {v8_acc - seed_acc:+.4f}')
    print(f'  v8 (edge recruit) vs matched top-k: {v8_acc - ctrl_acc:+.4f}  <-- THE ONE THAT MATTERS')
    print(f'  tsim: v8={v8_tsim:.3f}  top-k={ctrl_tsim:.3f}')
    if len(seeds) > 1:
        print(f'  (seed noise ~±{noise:.4f})')

    # Report edge structure
    v8_edges = [r['edges'] for r in res['v8_recruit'] if r['edges']]
    if v8_edges:
        avg_strong = np.mean([e['strong'] for e in v8_edges])
        avg_weak = np.mean([e['weak'] for e in v8_edges])
        avg_std = np.mean([e['std'] for e in v8_edges])
        print(f'  edge structure: {avg_strong:.0f} strong (>0.8), '
              f'{avg_weak:.0f} weak (<0.2), std={avg_std:.3f}')

    print()
    if v8_acc - ctrl_acc > thresh:
        print(f'  >>> v8 BEATS matched top-k by {v8_acc - ctrl_acc:+.4f}.')
        print(f'      Learned edge recruitment does something plain top-k cannot.')
        if v8_tsim < ctrl_tsim - 0.01:
            print(f'      AND tsim is lower — coalitions are more differentiated.')
    elif v8_acc - ctrl_acc < -thresh:
        print(f'  >>> Matched top-k WINS. Edge recruitment hurts even with a free graph.')
    else:
        print(f'  >>> No reliable difference. Even a free learned graph does not make')
        print(f'      recruitment beat plain top-k. The issue is the two-stage routing')
        print(f'      itself, not the graph mechanism.')


if __name__ == '__main__':
    main()
