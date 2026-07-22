"""v7: does forcing complementarity fix recruitment's redundancy problem?

THE DIAGNOSIS (from the v6 3-seed ablation):
  Recruitment lost to matched-k top-k (-1.6%, noise +/-3.1%) because it recruits
  the nodes MOST SIMILAR to the seed. Worse, it is self-reinforcing:
      proximity -> co-firing -> shared gradients -> same function
  Measured: tsim (how alike coalitions are across task types) rose to ~0.98 WITH
  recruitment vs ~0.81 without. Recruitment blurred specialization.

THE FIX:
  Penalise (position similarity x function similarity). Nodes that sit close
  must compute different things, so "near" comes to mean "complements me"
  instead of "same as me". Keeps the learned-position idea intact.

CONDITIONS (identical data/init per seed):
  C_matchedk   recruit OFF, k=K        -- the bar v6 failed to clear
  v6_baseline  recruit ON,  no penalty -- reproduces the known negative
  v7_c0.1 / c0.5 / c2.0                -- recruitment + complementarity, swept

WHAT WOULD COUNT AS WORKING (both, not one):
  1. tsim DROPS vs v6_baseline  -> the penalty actually differentiated nodes
  2. acc BEATS C_matchedk       -> that differentiation is USEFUL
  If (1) but not (2): diversity is real but useless -- still informative.
  If neither: the redundancy diagnosis was wrong.

Usage:
  python complementarity_test.py --seeds 3        # cheap config, ~5h CPU
  python complementarity_test.py --full --seeds 3 # GPU
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

# (label, recruit, complement_coef)  -- complement_coef None = penalty off
CONDITIONS = [
    ('C_matchedk',  False, None),
    ('v6_baseline', True,  None),
    ('v7_c0.1',     True,  0.1),
    ('v7_c0.5',     True,  0.5),
    ('v7_c2.0',     True,  2.0),
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
def mass_of(model, loader):
    model.eval()
    ms = []
    for x, _ in loader:
        for aux in model(x.to(DEVICE)).get('aux_data', []):
            if aux['type'] == 'coalition':
                ms.append(aux['node_activation'].sum(dim=1))
        break
    return torch.cat(ms).mean().item() if ms else 0.0


@torch.no_grad()
def type_similarity(model, datasets, n=800):
    """Max cosine sim between per-op-type coalition signatures. LOWER = more
    differentiated coalitions. This is the metric the fix should move."""
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


def run(label, recruit, ccoef, n_seeds, seed, datasets, tl, testl, genl):
    torch.manual_seed(seed)
    cfg = cfg_for(n_seeds)
    model = CoalitionModel(cfg).to(DEVICE)
    model.set_recruitment(recruit)
    model.set_norm_mode('sum')                 # magnitude-controlled, as in v6
    model.set_complement(ccoef is not None)

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
                    if ccoef is not None:
                        loss = loss + ccoef * aux['complement_penalty']
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.set_temperature(cfg['coalition']['temp_end'])

    acc, gen = evaluate(model, testl), evaluate(model, genl)
    mass, tsim = mass_of(model, testl), type_similarity(model, datasets)
    print(f'  seed{seed} {label:12s} | acc {acc:.4f}  gen {gen:.4f}  '
          f'mass {mass:.1f}  tsim {tsim:.3f}  [{time.time()-t0:.0f}s]', flush=True)
    return dict(acc=acc, gen=gen, mass=mass, tsim=tsim)


def main():
    base = cfg_for(4)
    datasets = create_datasets(base)
    tl = DataLoader(datasets['train'], batch_size=BS, shuffle=True)
    testl = DataLoader(datasets['test'], batch_size=256)
    genl = DataLoader(datasets['gen_test'], batch_size=256)

    print('=== v7: COMPLEMENTARITY TEST ===')
    print(f'device={DEVICE}  d={D_MODEL} L={N_LAYERS} {TRAIN_N} ex, {EPOCHS} ep, '
          f'{N_SEED_RUNS} seed(s)\n', flush=True)

    seeds = [1234 + 1000 * i for i in range(N_SEED_RUNS)]
    res = {c[0]: [] for c in CONDITIONS}

    for s in seeds:
        # v6's matched-k control used K = round(recruit-on mass). Reuse K=7,
        # the value the v6 3-seed run settled on, so the bar is identical.
        for label, recruit, ccoef in CONDITIONS:
            k = 7 if label == 'C_matchedk' else 4
            res[label].append(run(label, recruit, ccoef, k, s, datasets, tl, testl, genl))
        print('', flush=True)

    def agg(label, key):
        v = [r[key] for r in res[label]]
        return np.mean(v), (np.std(v) if len(v) > 1 else 0.0)

    print(f'{"condition":14s} {"test acc":>16} {"tsim":>16}')
    print('-' * 50)
    for label, _, _ in CONDITIONS:
        am, asd = agg(label, 'acc'); tm, tsd = agg(label, 'tsim')
        print(f'{label:14s} {am:>9.4f} ±{asd:.4f} {tm:>9.3f} ±{tsd:.3f}')

    ctrl = agg('C_matchedk', 'acc')[0]
    v6a, v6t = agg('v6_baseline', 'acc')[0], agg('v6_baseline', 'tsim')[0]
    noise = agg('C_matchedk', 'acc')[1] + agg('v6_baseline', 'acc')[1]
    thresh = max(0.01, noise)

    print(f'\n=== VERDICT ===')
    print(f'  v6 baseline vs matched top-k: {v6a-ctrl:+.4f}   (the known negative)')
    best, best_lbl = -9, None
    for label, _, cco in CONDITIONS:
        if cco is None:
            continue
        a, t = agg(label, 'acc')[0], agg(label, 'tsim')[0]
        print(f'  {label:10s} vs matched top-k: {a-ctrl:+.4f}   '
              f'tsim {t:.3f} (v6 {v6t:.3f}, {"DROPPED" if t < v6t - 0.01 else "no drop"})')
        if a - ctrl > best:
            best, best_lbl = a - ctrl, label
    if len(seeds) > 1:
        print(f'  (seed noise ~±{noise:.4f})')

    print()
    if best > thresh:
        print(f'  >>> {best_lbl} BEATS matched top-k by {best:+.4f}. The fix worked:')
        print(f'      forcing complementarity made recruitment do something top-k cannot.')
    elif any(agg(l, 'tsim')[0] < v6t - 0.01 for l, _, c in CONDITIONS if c is not None):
        print(f'  >>> tsim dropped but accuracy did not beat top-k. The penalty DID')
        print(f'      differentiate the nodes -- that differentiation just is not useful')
        print(f'      on this task. Diagnosis right, fix insufficient.')
    else:
        print(f'  >>> No tsim drop and no accuracy gain. The redundancy diagnosis')
        print(f'      does not hold, or the coefficient range is wrong.')


if __name__ == '__main__':
    main()
