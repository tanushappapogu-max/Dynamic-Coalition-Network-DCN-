"""Does chopping one FFN into more pieces cost more? The factorization curve.

The matched-capacity v9 run measured ONE point: at 16 nodes, splitting a wide FFN
into narrow blocks with a block-diagonal output layer cost -0.0973 token accuracy
with selection switched OFF. That single number invalidated the accuracy
conclusions of v6/v7/v8, but one point is not a claim.

This sweeps the number of pieces at FIXED TOTAL WIDTH:

    n_nodes    2     4     8    16    32
    d_node   512   256   128    64    32      (n_nodes * d_node = 1024 = d_ffn)

so every model has the same total hidden units and ~the same parameter count. The
only thing changing is how finely that width is chopped, and therefore how much
the block-diagonal output layer restricts mixing.

TWO CURVES COME OUT OF THIS
  structural tax   allon(n) - dense     cost of chopping into n pieces, no routing
  routing effect   subst(n) - allon(n)  what selection recovers at that granularity

Sparsity is held at 25% of nodes (budget = n_nodes/4) so the fraction of active
compute is constant and only granularity varies.

WHAT WOULD MAKE THIS A RESULT
  tax grows with n  -> finer chopping costs more, and since routing gain is
      bounded, there is an optimal granularity. That is a general statement about
      every architecture in this family, testable by anyone, and it explains a
      real pattern rather than proposing another mechanism.
  tax flat in n     -> the 16-node measurement was about something else
      (parameter count, or that specific width) and the story does not hold.

Usage
  python nodes_sweep.py --seeds 3
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.arithmetic import PAD_IDX
from src.models.coalition import CoalitionModel
from src.models.dense import DenseModel
from src.training.losses import coalition_load_balance_loss

N_SEED_RUNS = 1
TASK = 'latent'
for i, a in enumerate(sys.argv):
    if a == '--seeds':
        N_SEED_RUNS = int(sys.argv[i + 1])
    if a == '--task':
        TASK = sys.argv[i + 1]
assert TASK in ('latent', 'arithmetic'), TASK

# A second task is what makes the tax curve a general claim rather than a
# property of one synthetic dataset. Arithmetic is the natural control: it is the
# task the whole project used before, so its dense baseline is already known.
if TASK == 'arithmetic':
    from src.data.arithmetic import create_datasets
else:
    from src.data.latent_rule import create_datasets

# Same probe-validated config as latent_rule_test.py (dense = 0.571 token)
D_MODEL, N_LAYERS = 128, 2
TRAIN_N, EPOCHS, BS = 20000, 25, 128
TOTAL_WIDTH = 1024                    # == dense d_ffn
NODE_COUNTS = [2, 4, 8, 16, 32]
ACTIVE_FRACTION = 0.25                # budget = n_nodes * this
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def base_cfg():
    c = yaml.safe_load(open('configs/experiment.yaml'))
    if TASK == 'arithmetic':
        # Keep the yaml's arithmetic data schema, just shrink it to the same
        # budget as the latent task so the two sweeps cost the same.
        c['data'].update(train_size=TRAIN_N, val_size=2000, test_size=4000,
                         gen_test_size=2000)
        c['model'].update(d_model=D_MODEL, n_layers=N_LAYERS,
                          max_seq_len=c['data']['max_input_len'],
                          max_output_len=c['data']['max_output_len'])
    else:
        c['data'] = {
            'train_size': TRAIN_N, 'val_size': 2000, 'test_size': 4000,
            'seq_len': 5, 'max_input_len': 8, 'max_output_len': 6, 'seed': 42,
        }
        c['model'].update(d_model=D_MODEL, n_layers=N_LAYERS,
                          max_seq_len=8, max_output_len=6)
    c['dense']['d_ffn'] = TOTAL_WIDTH
    return c


def coalition_cfg(n_nodes, budget_mode):
    c = base_cfg()
    d_node = TOTAL_WIDTH // n_nodes
    c['coalition'].update(
        version=9, n_nodes=n_nodes, d_node=d_node,
        budget=max(1.0, n_nodes * ACTIVE_FRACTION),
        budget_mode=budget_mode,
    )
    return c, d_node


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    exact = total = tok_hit = tok_total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = model(x)['logits'].argmax(dim=-1)
        real = y != PAD_IDX
        tok_hit += ((p == y) & real).sum().item()
        tok_total += real.sum().item()
        exact += ((p == y) | ~real).all(dim=-1).sum().item()
        total += y.size(0)
    return exact / max(total, 1), tok_hit / max(tok_total, 1)


def train(model, cfg, tl, balance=True):
    rp = model.get_routing_params() if hasattr(model, 'get_routing_params') else []
    rid = {id(p) for p in rp}
    nrp = [p for p in model.parameters() if id(p) not in rid]
    groups = [{'params': nrp, 'lr': 3e-4}]
    if rp:
        groups.append({'params': rp, 'lr': 3e-3})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=3e-5)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    bc = cfg['coalition']['balance_coef']

    for ep in range(EPOCHS):
        model.train()
        if hasattr(model, 'set_temperature'):
            model.set_temperature(model.compute_temperature(ep, EPOCHS))
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = crit(out['logits'].reshape(-1, out['logits'].size(-1)), y.reshape(-1))
            if balance:
                for aux in out.get('aux_data', []):
                    if aux.get('type') == 'coalition':
                        a = aux['node_activation']
                        w = a / a.sum(dim=1, keepdim=True).clamp(min=1e-8)
                        loss = loss + bc * coalition_load_balance_loss(w)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    if hasattr(model, 'set_temperature'):
        model.set_temperature(cfg['coalition']['temp_end'])
    return model


def run_dense(seed, tl, testl):
    torch.manual_seed(seed)
    cfg = base_cfg()
    m = train(DenseModel(cfg).to(DEVICE), cfg, tl, balance=False)
    ex, tok = evaluate(m, testl)
    print(f'  seed{seed} dense              | tok {tok:.4f}  exact {ex:.4f}  '
          f'params {m.count_parameters():,}', flush=True)
    return tok


def run_coalition(n_nodes, mode, seed, tl, testl):
    torch.manual_seed(seed)
    cfg, d_node = coalition_cfg(n_nodes, mode)
    model = CoalitionModel(cfg).to(DEVICE)
    model.set_substitutive(True)
    t0 = time.time()
    model = train(model, cfg, tl, balance=(mode != 'uniform'))
    ex, tok = evaluate(model, testl)

    with torch.no_grad():
        mass = model(next(iter(testl))[0].to(DEVICE))['aux_data'][0][
            'node_activation'].sum(dim=1).mean().item()
    tag = 'allon' if mode == 'uniform' else 'subst'
    print(f'  seed{seed} n={n_nodes:<2d} d_node={d_node:<3d} {tag:5s} | '
          f'tok {tok:.4f}  exact {ex:.4f}  mass {mass:.1f}  '
          f'params {model.count_parameters():,}  [{time.time()-t0:.0f}s]', flush=True)
    return tok


def main():
    cfg = base_cfg()
    datasets = create_datasets(cfg)
    tl = DataLoader(datasets['train'], batch_size=BS, shuffle=True)
    testl = DataLoader(datasets['test'], batch_size=256)

    print('=== FACTORIZATION SWEEP: does finer chopping cost more? ===')
    print(f'task={TASK}  device={DEVICE}  d={D_MODEL} L={N_LAYERS} '
          f'{TRAIN_N} ex, {EPOCHS} ep, {N_SEED_RUNS} seed(s)')
    print(f'total hidden width held at {TOTAL_WIDTH}; only granularity varies')
    print(f'n_nodes: {NODE_COUNTS}   active fraction: {ACTIVE_FRACTION}\n', flush=True)

    seeds = [1234 + 1000 * i for i in range(N_SEED_RUNS)]
    dense_r, allon_r, subst_r = [], {n: [] for n in NODE_COUNTS}, {n: [] for n in NODE_COUNTS}

    for s in seeds:
        dense_r.append(run_dense(s, tl, testl))
        for n in NODE_COUNTS:
            allon_r[n].append(run_coalition(n, 'uniform', s, tl, testl))
            subst_r[n].append(run_coalition(n, 'fixed', s, tl, testl))
        print('', flush=True)

    def ms(v):
        return np.mean(v), (np.std(v) if len(v) > 1 else 0.0)

    dm, dsd = ms(dense_r)
    print(f'\ndense token: {dm:.4f} ±{dsd:.4f}\n')
    print(f'{"n_nodes":>7} {"d_node":>7} {"allon":>16} {"subst":>16} '
          f'{"TAX":>9} {"ROUTING":>9}')
    print('-' * 72)
    taxes, gains = [], []
    for n in NODE_COUNTS:
        am, asd = ms(allon_r[n]); sm, ssd = ms(subst_r[n])
        tax, gain = am - dm, sm - am
        taxes.append(tax); gains.append(gain)
        print(f'{n:>7} {TOTAL_WIDTH//n:>7} {am:>9.4f} ±{asd:.4f} '
              f'{sm:>9.4f} ±{ssd:.4f} {tax:>+9.4f} {gain:>+9.4f}')

    noise = dsd + max(ms(allon_r[n])[1] for n in NODE_COUNTS)
    print(f'\nseed noise ~±{noise:.4f}' if len(seeds) > 1 else '\n(single seed)')

    print('\n=== VERDICT ===')
    print(f'  tax at n=2 : {taxes[0]:+.4f}')
    print(f'  tax at n=32: {taxes[-1]:+.4f}')
    print(f'  change     : {taxes[-1] - taxes[0]:+.4f}')
    monotonic = all(taxes[i] >= taxes[i + 1] - 1e-9 for i in range(len(taxes) - 1))
    print(f'  monotonically worse with finer chopping: {monotonic}')

    best = int(np.argmax([g + t for g, t in zip(gains, taxes)]))
    print(f'\n  best net (tax+routing) at n_nodes={NODE_COUNTS[best]} '
          f'({taxes[best] + gains[best]:+.4f} vs dense)')

    print()
    if taxes[-1] - taxes[0] < -max(0.02, noise):
        print('  >>> The tax GROWS with granularity. Finer chopping costs more, and')
        print('      since routing recovers a bounded amount there is an optimal')
        print('      number of experts. That is a general claim about this family of')
        print('      architectures, and it explains underperformance rather than')
        print('      proposing yet another mechanism.')
    elif abs(taxes[-1] - taxes[0]) < max(0.02, noise):
        print('  >>> The tax is FLAT in granularity. The -0.0973 at n=16 was not')
        print('      about chopping -- it is a property of that width or of the')
        print('      coalition FFN in general. The curve story does not hold.')
    else:
        print('  >>> The tax SHRINKS with finer chopping -- opposite of predicted.')
        print('      Worth understanding before any writeup.')


if __name__ == '__main__':
    main()
