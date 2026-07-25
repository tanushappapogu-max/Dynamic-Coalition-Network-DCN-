"""v9: do mass-conserving coalitions beat routing when the rule is hidden?

TASK (src/data/latent_rule.py): four algorithms over a digit string, selected by
(sum of digits) mod 4 -- nonlinear, so a linear gate cannot read it.

CONFIG IS PROBE-VALIDATED. probe_learnability.py at the default config below:
    reverse 1.000 / sort 1.000 / prefixsum 0.694 token  (single rules, no selector)
    latent  0.571 token                                 (dense, full task)
So the achievable bar is ~0.92 token and dense sits at 0.571 -- roughly 35 points
of headroom, neither floor nor ceiling. That window is why this task is usable and
arithmetic was not.

PRIMARY METRIC IS TOKEN ACCURACY. Exact-match on 5 digits reports prefixsum as
~0.009 when the model has in fact learned ~0.69 of it; brittle metrics at a floor
are how v6-v8 burned time. Exact match is reported as secondary.

CONDITIONS
  dense        no routing -- the bar that beat every DCN so far
  moe_linear   top-k MoE, linear gate (structurally cannot compute mod 4)
  moe_mlp      top-k MoE, hidden layer in the gate, so the baseline CAN express a
               nonlinear selector. Without this a DCN win would only mean
               "nonlinear beats linear routing", which is not about coalitions.
  v9_add       resonance drive, mass FREE to grow -- the v6-v8 behaviour
  v9_subst     resonance drive, mass PINNED to 4 -- join by displacing
  v9_nobal     v9_subst with balance_coef=0. Load balancing pushes every node to
               1/16 usage, but the probe measured the four rules at wildly
               different difficulty (prefixsum 0.694 vs reverse 1.000), so even
               allocation is probably the WRONG prior. This asks what allocation
               emerges when nothing forces it.
  v9_drive     v9_subst with the mass set to the participation ratio of the drive
               vector, restoring variable coalition size -- read off the response
               rather than fixed or predicted.

v9_add vs v9_subst is the isolation: same drive, same coupling, the only
difference is whether activation can accumulate.

NOTE on the one asymmetry: v9_add gets a size loss and the others do not. Without
it the additive control collapses to mass 0 (measured -- gradient descent switches
the FFN off entirely), which would make the ablation vacuous. It is the minimum
needed to keep the control alive, and it targets the same mass the others use.

Usage
  python latent_rule_test.py --quick            # smoke, tiny
  python latent_rule_test.py --seeds 3          # probe-validated config
  python latent_rule_test.py --full --seeds 3   # Colab GPU
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

from src.data.arithmetic import PAD_IDX
from src.data.latent_rule import create_datasets, classify_rule, RULE_NAMES, N_RULES
from src.models.coalition import CoalitionModel
from src.models.dense import DenseModel
from src.models.moe import MoEModel
from src.training.losses import (
    coalition_load_balance_loss, coalition_size_loss, moe_load_balance_loss,
)

QUICK = '--quick' in sys.argv
FULL = '--full' in sys.argv
N_SEED_RUNS = 1
for i, a in enumerate(sys.argv):
    if a == '--seeds':
        N_SEED_RUNS = int(sys.argv[i + 1])

if FULL:
    D_MODEL, N_LAYERS, D_NODE = 256, 4, 64
    TRAIN_N, EPOCHS, BS = 60000, 40, 128
elif QUICK:
    D_MODEL, N_LAYERS, D_NODE = 96, 2, 32
    TRAIN_N, EPOCHS, BS = 6000, 12, 128
else:                       # probe-validated: dense = 0.571 token here
    # d_node=64 so 16 nodes x 64 = 1024 = dense's d_ffn. The earlier run used 48
    # (=768 hidden), handing the coalition models a 25% width deficit that got
    # misread as a routing result. Match the width; the block-diagonal second
    # layer remains, which is what v9_allon is for.
    D_MODEL, N_LAYERS, D_NODE = 128, 2, 64
    TRAIN_N, EPOCHS, BS = 20000, 25, 128

BUDGET = 4.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# label -> (kind, balance_coef_scale, budget_mode, substitutive, size_loss)
SPEC = {
    'dense':      ('dense', 0.0, None,      None,  False),
    'moe_linear': ('moe',   1.0, None,      None,  False),
    'moe_mlp':    ('moe',   1.0, None,      None,  False),
    'v9_allon':   ('v9',    0.0, 'uniform', True,  False),
    'v9_add':     ('v9',    1.0, 'fixed',   False, True),
    'v9_subst':   ('v9',    1.0, 'fixed',   True,  False),
    'v9_nobal':   ('v9',    0.0, 'fixed',   True,  False),
    'v9_drive':   ('v9',    1.0, 'drive',   True,  False),
}
# v9_allon is the reference the v9 variants must be read against, NOT dense:
# it is the same block-structured FFN with selection switched off, so
# (v9_allon - dense) is the structural tax and (v9_subst - v9_allon) is what
# routing actually contributes.
CONDITIONS = (['dense', 'v9_allon', 'v9_subst'] if QUICK
              else ['dense', 'moe_linear', 'moe_mlp', 'v9_allon',
                    'v9_add', 'v9_subst', 'v9_nobal', 'v9_drive'])


def base_cfg():
    c = yaml.safe_load(open('configs/experiment.yaml'))
    c['data'] = {
        'train_size': TRAIN_N, 'val_size': 2000, 'test_size': 4000,
        'seq_len': 5, 'max_input_len': 8, 'max_output_len': 6, 'seed': 42,
    }
    c['model'].update(d_model=D_MODEL, n_layers=N_LAYERS,
                      max_seq_len=8, max_output_len=6)
    c['coalition']['d_node'] = D_NODE
    c['moe'].update(d_expert=D_NODE, top_k=4)     # matched to the coalition budget
    return c


def build(label):
    kind, _, budget_mode, subst, _ = SPEC[label]
    cfg = base_cfg()
    if kind == 'dense':
        return DenseModel(cfg), cfg
    if kind == 'moe':
        cfg['moe']['gate_hidden'] = D_MODEL if label == 'moe_mlp' else None
        return MoEModel(cfg), cfg
    cfg['coalition'].update(version=9, budget=BUDGET, budget_mode=budget_mode)
    model = CoalitionModel(cfg)
    model.set_substitutive(subst)
    return model, cfg


@torch.no_grad()
def evaluate(model, loader):
    """(exact-match, token) -- token is primary."""
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


@torch.no_grad()
def per_rule_token(model, dataset):
    """Token accuracy per rule -- shows lopsided difficulty and lopsided skill."""
    model.eval()
    hit = defaultdict(int); tot = defaultdict(int)
    idx = 0
    for x, y in DataLoader(dataset, batch_size=256):
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = model(x)['logits'].argmax(dim=-1)
        real = (y != PAD_IDX)
        ok = ((p == y) & real).sum(dim=-1).cpu().numpy()
        cnt = real.sum(dim=-1).cpu().numpy()
        for j in range(len(ok)):
            r = classify_rule(dataset.data[idx + j][0])
            hit[r] += int(ok[j]); tot[r] += int(cnt[j])
        idx += len(ok)
    return {r: hit[r] / max(tot[r], 1) for r in RULE_NAMES}


def _usage_vector(aux, batch):
    """Per-sequence node/expert usage, comparable across architectures.

    Coalitions give this directly. For MoE we pool the per-token gate
    distribution over the sequence, so "did routing line up with the four rules"
    is answerable for the BASELINE too -- otherwise decodability would only be
    reported for the architecture we are advocating, which proves nothing.
    """
    kind = aux.get('type')
    if kind == 'coalition':
        return aux['node_activation'].float(), True
    if kind == 'moe':
        gp = aux['gate_probs'].float()
        return gp.view(batch, -1, gp.size(-1)).mean(dim=1), False
    return None, False


@torch.no_grad()
def routing_report(model, dataset, n=2000):
    """Nearest-centroid rule decodability (chance 0.25), plus how mass is spent.

    Decodability is the real claim: does coalition COMPOSITION track the subtask.
    It is indifferent to whether usage is even -- a model that consistently
    spends 6 nodes on prefixsum and 2 on increment is a success, not an
    imbalance. The probe measured the four rules at unequal difficulty, so even
    allocation is probably the wrong thing to want.
    """
    model.eval()
    nan = dict(decode=float('nan'), sim=float('nan'), mass=float('nan'),
               mass_std=float('nan'), usage_gini=float('nan'))
    n = min(n, len(dataset))
    acts, labels, masses = [], [], []
    has_mass = False
    idx = 0
    for x, _ in DataLoader(Subset(dataset, range(n)), batch_size=256):
        x = x.to(DEVICE)
        aux = model(x).get('aux_data')
        if not aux:
            return nan
        vec, has_mass = _usage_vector(aux[0], x.size(0))
        if vec is None:
            return nan
        a = vec.cpu().numpy()
        masses.append(a.sum(axis=1))
        for j in range(a.shape[0]):
            acts.append(a[j])
            labels.append(RULE_NAMES.index(classify_rule(dataset.data[idx + j][0])))
        idx += a.shape[0]

    A = np.stack(acts); L = np.array(labels)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    half = len(An) // 2
    cents = []
    for r in range(N_RULES):
        sel = An[:half][L[:half] == r]
        cents.append(sel.mean(axis=0) if len(sel) else np.zeros(An.shape[1]))
    Cn = np.stack(cents)
    Cn = Cn / (np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-9)
    decode = float(((An[half:] @ Cn.T).argmax(axis=1) == L[half:]).mean())
    sims = [float(Cn[i] @ Cn[j]) for i in range(N_RULES) for j in range(i + 1, N_RULES)]

    mass = np.concatenate(masses)
    # Gini of per-node usage: 0 = perfectly even, ->1 = a few nodes hog it.
    use = np.sort(A.mean(axis=0))
    k = len(use)
    gini = float((2 * np.arange(1, k + 1) - k - 1).dot(use) / (k * use.sum() + 1e-9))
    # MoE gate probs are softmax-normalised, so their "mass" is always 1 and not
    # comparable to a coalition's -- report it only where it means something.
    return dict(decode=decode, sim=float(np.mean(sims)),
                mass=float(mass.mean()) if has_mass else float('nan'),
                mass_std=float(mass.std()) if has_mass else float('nan'),
                usage_gini=gini)


def run(label, seed, datasets, tl, testl):
    torch.manual_seed(seed)
    model, cfg = build(label)
    model = model.to(DEVICE)
    _, bal_scale, _, _, use_size_loss = SPEC[label]

    rp = model.get_routing_params() if hasattr(model, 'get_routing_params') else []
    rid = {id(p) for p in rp}
    nrp = [p for p in model.parameters() if id(p) not in rid]
    groups = [{'params': nrp, 'lr': 3e-4}]
    if rp:
        groups.append({'params': rp, 'lr': 3e-3})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=3e-5)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    cc, mc = cfg['coalition'], cfg['moe']

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
                if aux.get('type') == 'coalition':
                    a = aux['node_activation']
                    if bal_scale:
                        w = a / a.sum(dim=1, keepdim=True).clamp(min=1e-8)
                        loss = loss + bal_scale * cc['balance_coef'] * \
                            coalition_load_balance_loss(w)
                    if use_size_loss:
                        loss = loss + cc['size_coef'] * coalition_size_loss(a, BUDGET)
                elif aux.get('type') == 'moe':
                    loss = loss + mc['balance_coef'] * moe_load_balance_loss(
                        aux['gate_probs'], aux['top_k_indices'], mc['n_experts'])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    if hasattr(model, 'set_temperature'):
        model.set_temperature(cfg['coalition']['temp_end'])

    ex, tok = evaluate(model, testl)
    rep = routing_report(model, datasets['test'])
    pr = per_rule_token(model, datasets['test'])

    print(f'  seed{seed} {label:11s} | tok {tok:.4f}  exact {ex:.4f}  '
          f'decode {rep["decode"]:.3f}  mass {rep["mass"]:.1f}'
          f'±{rep["mass_std"]:.1f}  gini {rep["usage_gini"]:.2f}  '
          f'[{time.time()-t0:.0f}s]', flush=True)
    print(f'{"":21s}per-rule tok: ' +
          '  '.join(f'{r[:5]}={pr[r]:.2f}' for r in RULE_NAMES), flush=True)
    return dict(tok=tok, exact=ex, per_rule=pr,
                params=model.count_parameters(), **rep)


def main():
    cfg = base_cfg()
    datasets = create_datasets(cfg)
    tl = DataLoader(datasets['train'], batch_size=BS, shuffle=True)
    testl = DataLoader(datasets['test'], batch_size=256)

    print('=== v9: LATENT-RULE TASK ===')
    print(f'device={DEVICE}  d={D_MODEL} L={N_LAYERS} {TRAIN_N} ex, {EPOCHS} ep, '
          f'{N_SEED_RUNS} seed(s)')
    print(f'primary metric: TOKEN accuracy   (probe: dense 0.571, achievable ~0.92)')
    print(f'conditions: {", ".join(CONDITIONS)}\n', flush=True)

    seeds = [1234 + 1000 * i for i in range(N_SEED_RUNS)]
    res = {c: [] for c in CONDITIONS}
    for s in seeds:
        for label in CONDITIONS:
            res[label].append(run(label, s, datasets, tl, testl))
        print('', flush=True)

    def agg(label, key):
        v = [r[key] for r in res[label]]
        return np.mean(v), (np.std(v) if len(v) > 1 else 0.0)

    print(f'\n{"condition":12s} {"token":>16} {"exact":>10} {"decode":>8} '
          f'{"mass":>12} {"gini":>6}')
    print('-' * 70)
    for label in CONDITIONS:
        tm, tsd = agg(label, 'tok')
        print(f'{label:12s} {tm:>9.4f} ±{tsd:.4f} {agg(label,"exact")[0]:>10.4f} '
              f'{agg(label,"decode")[0]:>8.3f} '
              f'{agg(label,"mass")[0]:>6.1f}±{agg(label,"mass_std")[0]:<5.1f} '
              f'{agg(label,"usage_gini")[0]:>6.2f}')

    sub = agg('v9_subst', 'tok')[0]
    add = agg('v9_add', 'tok')[0]
    noise = agg('v9_subst', 'tok')[1] + agg('dense', 'tok')[1]
    thresh = max(0.01, noise)

    print('\n=== VERDICT ===')
    if 'v9_allon' in res:
        allon = agg('v9_allon', 'tok')[0]
        dn = agg('dense', 'tok')[0]
        print('  DECOMPOSITION (the thing every earlier version confounded):')
        print(f'    structural tax   v9_allon - dense    : {allon - dn:+.4f}')
        print(f'      cost of 16 block FFNs w/ block-diagonal output vs one wide')
        print(f'      dense FFN, with selection switched OFF. Nothing to do with routing.')
        print(f'    routing effect   v9_subst - v9_allon : {sub - allon:+.4f}')
        print(f'      what mass-conserving selection actually buys, tax removed.')
        print()
    for other in CONDITIONS:
        if other != 'v9_subst':
            print(f'  v9_subst vs {other:11s}: {sub - agg(other, "tok")[0]:+.4f}')
    if len(seeds) > 1:
        print(f'  (seed noise ~±{noise:.4f})')
    print(f'\n  substitution vs accumulation: {sub - add:+.4f}   <-- the v9 claim')
    print(f'  rule decodability: subst {agg("v9_subst","decode")[0]:.3f}  '
          f'add {agg("v9_add","decode")[0]:.3f}  '
          f'nobal {agg("v9_nobal","decode")[0] if "v9_nobal" in res else float("nan"):.3f}'
          f'   (chance 0.250)')

    print()
    rivals = [o for o in CONDITIONS if o != 'v9_subst']
    if all(sub - agg(o, 'tok')[0] > thresh for o in rivals):
        print('  >>> v9_subst beats every baseline. Mass-conserving coalitions do')
        print('      something neither dense nor routing can.')
    elif sub - add > thresh:
        print('  >>> Substitution beats accumulation but not the baselines. The mass')
        print('      argument holds; the architecture is still not better than')
        print('      routing well.')
    else:
        print('  >>> No advantage. Check decode: near 0.250 means coalitions never')
        print('      aligned with the four real subtasks, so the selection mechanism')
        print('      is the problem rather than the mass rule.')


if __name__ == '__main__':
    main()
