"""THE experiment: does proximity recruitment actually do anything?

Stage 1 of the DCN (seeds via learned keys) is standard routing — everyone has
it. Stage 2 (nodes recruit their neighbours in a learned position space) is the
contribution. It has never been switched off and compared.

CONDITIONS (identical data / init / epochs per seed):

  A_orig       recruit ON,  k=4   norm=n_seeds  — v5 exactly as shipped
  A_full       recruit ON,  k=4   norm=sum      — magnitude-controlled
  B_seedsonly  recruit OFF, k=4   norm=sum      — same seeds, no neighbours
  C_matchedk   recruit OFF, k=K   norm=sum      — K matches A_full's activation mass

C IS THE CONTROL THAT MATTERS. Beating B is trivial (more nodes active). The
real question is whether recruitment beats plain top-K *at equal compute*. If
A_full ~= C_matchedk, recruitment is just an expensive way to raise k.

Magnitude control: v5 divides by n_seeds while activating more than n_seeds
worth of mass, so recruitment also makes the output louder. norm='sum' holds
total weight at 1 so we measure node CHOICE, not loudness.

Usage:
  python recruitment_ablation.py                 # local smoke (small, 1 seed)
  python recruitment_ablation.py --full --seeds 3   # T4/GPU: real config, error bars
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
N_SEEDS_RUNS = 1
for i, a in enumerate(sys.argv):
    if a == '--seeds':
        N_SEEDS_RUNS = int(sys.argv[i + 1])

if FULL:                      # GPU config — real accuracy, comparable to the 89.87% run
    D_MODEL, N_LAYERS, D_NODE = 256, 4, 64
    TRAIN_N, EPOCHS, BS = 80000, 60, 128
else:                         # CPU smoke config
    D_MODEL, N_LAYERS, D_NODE = 160, 3, 48
    TRAIN_N, EPOCHS, BS = 25000, 50, 128

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    """(activation MASS, recruited mass). Mass not >0.1 count: recruitment
    spreads soft activation everywhere, so a count reports 16/16 and would make
    the matched-k control ask for the whole network."""
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
    """Max cosine sim between per-op-type coalition signatures.
    <1.0 => different ops recruit different coalitions."""
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


def run(label, n_seeds, recruit, norm_mode, seed, datasets, tl, testl, genl, quiet=False):
    torch.manual_seed(seed)
    cfg = cfg_for(n_seeds)
    model = CoalitionModel(cfg).to(DEVICE)
    model.set_recruitment(recruit)
    model.set_norm_mode(norm_mode)

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
    if not quiet:
        print(f'  seed{seed} {label:12s} k={n_seeds} recruit={str(recruit):5s} | '
              f'acc {acc:.4f}  gen {gen:.4f}  mass {mass:.1f}  recr {recr:.2f}  '
              f'tsim {tsim:.3f}  [{time.time()-t0:.0f}s]', flush=True)
    return dict(acc=acc, gen=gen, mass=mass, tsim=tsim)


def main():
    base = cfg_for(4)
    datasets = create_datasets(base)
    tl = DataLoader(datasets['train'], batch_size=BS, shuffle=True)
    testl = DataLoader(datasets['test'], batch_size=256)
    genl = DataLoader(datasets['gen_test'], batch_size=256)

    print(f'=== v6: RECRUITMENT ABLATION (v5 architecture, seed-verified) ===')
    print(f'device={DEVICE}  d={D_MODEL} L={N_LAYERS} {TRAIN_N} ex, {EPOCHS} ep, '
          f'{N_SEEDS_RUNS} seed(s)\n', flush=True)
    if N_SEEDS_RUNS < 3:
        print('NOTE: run with --seeds 3 for the verified result (this is the point of v6).\n', flush=True)

    seeds = [1234 + 1000 * i for i in range(N_SEEDS_RUNS)]
    acc = {k: [] for k in ('A_orig', 'A_full', 'B_seedsonly', 'C_matchedk')}
    gen = {k: [] for k in acc}
    extra = {}

    for s in seeds:
        r = run('A_orig', 4, True, 'n_seeds', s, datasets, tl, testl, genl)
        acc['A_orig'].append(r['acc']); gen['A_orig'].append(r['gen'])

        a = run('A_full', 4, True, 'sum', s, datasets, tl, testl, genl)
        acc['A_full'].append(a['acc']); gen['A_full'].append(a['gen'])
        extra['mass'] = a['mass']; extra['tsim'] = a['tsim']

        b = run('B_seedsonly', 4, False, 'sum', s, datasets, tl, testl, genl)
        acc['B_seedsonly'].append(b['acc']); gen['B_seedsonly'].append(b['gen'])

        K = max(2, min(base['coalition']['n_nodes'], int(round(a['mass']))))
        c = run('C_matchedk', K, False, 'sum', s, datasets, tl, testl, genl)
        acc['C_matchedk'].append(c['acc']); gen['C_matchedk'].append(c['gen'])
        print('', flush=True)

    def ms(v):
        return np.mean(v), (np.std(v) if len(v) > 1 else 0.0)

    print(f'{"condition":14s} {"test acc":>16} {"gen acc":>16}')
    print('-' * 48)
    for k in ('A_orig', 'A_full', 'B_seedsonly', 'C_matchedk'):
        am, asd = ms(acc[k]); gm, gsd = ms(gen[k])
        print(f'{k:14s} {am:>9.4f} ±{asd:.4f} {gm:>9.4f} ±{gsd:.4f}')

    da = np.mean(acc['A_full']) - np.mean(acc['B_seedsonly'])
    dc = np.mean(acc['A_full']) - np.mean(acc['C_matchedk'])
    sd = np.std(acc['A_full']) + np.std(acc['C_matchedk'])

    print(f'\n=== VERDICT ===')
    print(f'  A_full mass {extra.get("mass",0):.1f} -> matched-k control used K='
          f'{max(2, min(16, int(round(extra.get("mass",4)))))}')
    print(f'  recruitment vs seeds-only:  {da:+.4f}')
    print(f'  recruitment vs matched top-k: {dc:+.4f}   <-- THE ONE THAT MATTERS')
    if len(seeds) > 1:
        print(f'  (combined seed noise ~±{sd:.4f})')
    if dc > max(0.01, sd):
        print('  >>> Recruitment BEATS plain top-k at equal compute.')
        print('      The proximity structure does something top-k cannot. Real result.')
    elif dc < -max(0.01, sd):
        print('  >>> Plain top-k WINS. Recruitment actively hurts.')
    else:
        print('  >>> No reliable difference. Recruitment is an expensive way to')
        print('      raise k; plain top-k gets the same for free. Honest negative.')


if __name__ == '__main__':
    main()
