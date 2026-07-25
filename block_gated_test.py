"""Does block-gating beat the mixture formulation at equal sparsity?

THE ARCHITECTURE (src/models/gated_block.py)
    output = W2 @ (mask ⊙ GELU(W1 @ x))
One wide FFN, hidden units gated in blocks, output projection full and shared.
Verified: with every block on this is EXACTLY dense (max diff 0.000e+00 at
n_blocks 2..32). A convex mixture cannot be -- its output is 1/n the magnitude
(measured 0.0625 at n=16), which is the attenuation that sank v6-v9.

CONDITIONS (matched d_ffn=1024, so matched FFN parameters throughout)
  dense       reference
  bg_all      all 16 blocks on -- must land at dense; sanity on the whole rig
  bg_k8/k4/k2 8, 4, 2 of 16 blocks -- 50%, 25%, 12.5% of FFN matmul work
  moe_linear  standard top-k MoE, 4 of 16 experts, convex-normalised gate
  v9_subst    our mass-conserving mixture, 4 of 16 -- also convex-normalised

WHAT WOULD MAKE THIS NOVEL AND TRUE
  bg_k4 >= dense at 25% of the FFN work, while moe_linear and v9_subst sit below
  it at the same sparsity. That isolates the claim to the COMBINATION RULE --
  every condition has the same parameter budget, the same task, and comparable
  selection; the only structural difference is whether active units are summed
  through a shared projection or averaged through private ones.

  If bg_k4 also lands below dense, the attenuation story is incomplete and
  something else is limiting sparse layers here.

Usage
  python block_gated_test.py --seeds 1        # scout
  python block_gated_test.py --seeds 3        # error bars
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.arithmetic import PAD_IDX
from src.data.latent_rule import create_datasets
from src.models.coalition import CoalitionModel
from src.models.dense import DenseModel
from src.models.gated_block import BlockGatedModel, assert_equals_dense
from src.models.moe import MoEModel
from src.training.losses import coalition_load_balance_loss, moe_load_balance_loss

FULL = '--full' in sys.argv
N_SEED_RUNS = 1
for i, a in enumerate(sys.argv):
    if a == '--seeds':
        N_SEED_RUNS = int(sys.argv[i + 1])

if FULL:                    # GPU: does the result survive a real size?
    D_MODEL, N_LAYERS = 256, 4
    D_FFN, N_BLOCKS = 2048, 16
    TRAIN_N, EPOCHS, BS = 80000, 40, 128
else:
    D_MODEL, N_LAYERS = 128, 2
    D_FFN, N_BLOCKS = 1024, 16
    TRAIN_N, EPOCHS, BS = 20000, 25, 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# label -> (kind, k_active)
SPEC = {
    'dense':      ('dense', None),
    'bg_all':     ('bg', 16),
    'bg_k8':      ('bg', 8),
    'bg_k4':      ('bg', 4),
    'bg_k2':      ('bg', 2),
    'moe_linear': ('moe', 4),
    'v9_subst':   ('v9', 4),
}
CONDITIONS = ['dense', 'bg_all', 'bg_k8', 'bg_k4', 'bg_k2',
              'moe_linear', 'v9_subst']


def base_cfg():
    c = yaml.safe_load(open('configs/experiment.yaml'))
    c['data'] = {
        'train_size': TRAIN_N, 'val_size': 2000, 'test_size': 4000,
        'seq_len': 5, 'max_input_len': 8, 'max_output_len': 6, 'seed': 42,
    }
    c['model'].update(d_model=D_MODEL, n_layers=N_LAYERS,
                      max_seq_len=8, max_output_len=6)
    c['dense']['d_ffn'] = D_FFN
    c['block_gated'] = {
        'd_ffn': D_FFN, 'n_blocks': N_BLOCKS, 'k_active': N_BLOCKS,
        'rescale': 'inv', 'select': 'resonance',
        'temp_start': 1.0, 'temp_end': 0.1, 'warmup_fraction': 0.1,
        'balance_coef': 0.01,
    }
    # MoE and v9 get the same total FFN width, split 16 ways
    c['moe'].update(n_experts=N_BLOCKS, d_expert=D_FFN // N_BLOCKS, top_k=4,
                    gate_hidden=None)
    c['coalition'].update(n_nodes=N_BLOCKS, d_node=D_FFN // N_BLOCKS,
                          version=9, budget=4.0, budget_mode='fixed')
    return c


def build(label):
    kind, k = SPEC[label]
    cfg = base_cfg()
    if kind == 'dense':
        return DenseModel(cfg), cfg
    if kind == 'moe':
        return MoEModel(cfg), cfg
    if kind == 'v9':
        m = CoalitionModel(cfg); m.set_substitutive(True)
        return m, cfg
    cfg['block_gated']['k_active'] = k
    cfg['block_gated']['select'] = 'all' if k >= N_BLOCKS else 'resonance'
    return BlockGatedModel(cfg), cfg


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ex = tot = th = tt = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = model(x)['logits'].argmax(dim=-1)
        real = y != PAD_IDX
        th += ((p == y) & real).sum().item(); tt += real.sum().item()
        ex += ((p == y) | ~real).all(dim=-1).sum().item(); tot += y.size(0)
    return ex / max(tot, 1), th / max(tt, 1)


def run(label, seed, tl, testl):
    torch.manual_seed(seed)
    model, cfg = build(label)
    model = model.to(DEVICE)

    rp = model.get_routing_params() if hasattr(model, 'get_routing_params') else []
    rid = {id(p) for p in rp}
    nrp = [p for p in model.parameters() if id(p) not in rid]
    groups = [{'params': nrp, 'lr': 3e-4}]
    if rp:
        groups.append({'params': rp, 'lr': 3e-3})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=3e-5)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        if hasattr(model, 'set_temperature'):
            model.set_temperature(model.compute_temperature(ep, EPOCHS))
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = crit(out['logits'].reshape(-1, out['logits'].size(-1)), y.reshape(-1))
            for aux in out.get('aux_data', []):
                t = aux.get('type')
                if t == 'block_gated':
                    m = aux['node_activation']
                    if m.std() > 0:      # only meaningful when actually gating
                        loss = loss + cfg['block_gated']['balance_coef'] * \
                            coalition_load_balance_loss(m / m.sum(dim=1, keepdim=True).clamp(min=1e-8))
                elif t == 'coalition':
                    a = aux['node_activation']
                    loss = loss + cfg['coalition']['balance_coef'] * \
                        coalition_load_balance_loss(a / a.sum(dim=1, keepdim=True).clamp(min=1e-8))
                elif t == 'moe':
                    loss = loss + cfg['moe']['balance_coef'] * moe_load_balance_loss(
                        aux['gate_probs'], aux['top_k_indices'], cfg['moe']['n_experts'])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    if hasattr(model, 'set_temperature'):
        model.set_temperature(0.1)

    ex, tok = evaluate(model, testl)
    ff = model.flops_fraction() if hasattr(model, 'flops_fraction') else (
        SPEC[label][1] / N_BLOCKS if SPEC[label][1] else 1.0)
    print(f'  seed{seed} {label:11s} | tok {tok:.4f}  exact {ex:.4f}  '
          f'ffn_flops {ff*100:>5.1f}%  params {model.count_parameters():,}  '
          f'[{time.time()-t0:.0f}s]', flush=True)
    return dict(tok=tok, exact=ex, flops=ff, params=model.count_parameters())


def main():
    print('=== BLOCK-GATED FFN vs MIXTURE FORMULATIONS ===')
    d = assert_equals_dense(d_model=D_MODEL, d_ffn=D_FFN, n_blocks=N_BLOCKS)
    print(f'precondition: block-gated(all on) == dense, max diff {d:.3e}  OK')
    print(f'device={DEVICE}  d={D_MODEL} L={N_LAYERS} d_ffn={D_FFN} '
          f'n_blocks={N_BLOCKS}  {TRAIN_N} ex, {EPOCHS} ep, {N_SEED_RUNS} seed(s)\n',
          flush=True)

    cfg = base_cfg()
    ds = create_datasets(cfg)
    tl = DataLoader(ds['train'], batch_size=BS, shuffle=True)
    testl = DataLoader(ds['test'], batch_size=256)

    seeds = [1234 + 1000 * i for i in range(N_SEED_RUNS)]
    res = {c: [] for c in CONDITIONS}
    for s in seeds:
        for label in CONDITIONS:
            res[label].append(run(label, s, tl, testl))
        print('', flush=True)

    def agg(l, k):
        v = [r[k] for r in res[l]]
        return np.mean(v), (np.std(v) if len(v) > 1 else 0.0)

    print(f'\n{"condition":12s} {"token":>16} {"exact":>9} {"ffn flops":>11} {"params":>10}')
    print('-' * 64)
    for l in CONDITIONS:
        tm, ts = agg(l, 'tok')
        print(f'{l:12s} {tm:>9.4f} ±{ts:.4f} {agg(l,"exact")[0]:>9.4f} '
              f'{agg(l,"flops")[0]*100:>10.1f}% {res[l][0]["params"]:>10,}')

    dn = agg('dense', 'tok')[0]
    noise = agg('dense', 'tok')[1] + agg('bg_k4', 'tok')[1]
    thresh = max(0.01, noise)

    print('\n=== VERDICT ===')
    print(f'  sanity   bg_all  - dense : {agg("bg_all","tok")[0]-dn:+.4f} '
          f'(same function; large gap here means the rig is wrong)')
    for l in ('bg_k8', 'bg_k4', 'bg_k2'):
        print(f'  {l:8s} vs dense       : {agg(l,"tok")[0]-dn:+.4f} '
              f'at {agg(l,"flops")[0]*100:.1f}% FFN flops')
    print()
    print(f'  SAME SPARSITY (4 of 16), combination rule is the only difference:')
    for l in ('bg_k4', 'moe_linear', 'v9_subst'):
        print(f'    {l:11s} {agg(l,"tok")[0]:.4f}')
    bg4 = agg('bg_k4', 'tok')[0]
    print(f'    bg_k4 - moe_linear : {bg4-agg("moe_linear","tok")[0]:+.4f}')
    print(f'    bg_k4 - v9_subst   : {bg4-agg("v9_subst","tok")[0]:+.4f}')
    if len(seeds) > 1:
        print(f'  (seed noise ~±{noise:.4f})')

    print()
    beats_mix = (bg4 - agg('moe_linear', 'tok')[0] > thresh and
                 bg4 - agg('v9_subst', 'tok')[0] > thresh)
    if bg4 - dn > -thresh and beats_mix:
        print('  >>> Block-gating holds dense accuracy at 25% of FFN flops AND beats')
        print('      both mixture formulations at identical sparsity. The combination')
        print('      rule was the binding constraint, not the router.')
    elif beats_mix:
        print('  >>> Block-gating beats both mixtures at equal sparsity but does not')
        print('      reach dense. The attenuation account is right and incomplete.')
    else:
        print('  >>> Block-gating does not beat the mixtures. The 1/n attenuation is')
        print('      real in the forward pass but is not what limits trained accuracy.')


if __name__ == '__main__':
    main()
