"""Is the latent-rule task learnable at all, and which part is hard?

The v9 pilot put dense at 3.4% -- a floor. Comparing architectures at a floor is
what wasted v6-v8, so before spending GPU time, find out WHERE the difficulty is.

LADDER (dense only, identical config at every rung):
  reverse     one algorithm, no selector   -- positional remap
  sort        one algorithm, no selector   -- global comparison
  prefixsum   one algorithm, no selector   -- sequential scan
  latent      all four + hidden mod-4 selector  -- the real task

READING IT:
  single rules fail  -> the ARCHITECTURE is the bottleneck. The output head reads
                        a mean-pooled sequence, so order-dependent outputs may not
                        survive pooling. Task needs redesign, not a bigger model.
  single rules work,
  latent fails       -> the SELECTOR is the hard part. That is the interesting
                        regime and the one the coalition mechanism is meant for.
  everything works   -> raise difficulty until dense stops acing it, then compare.

Reports token accuracy as well as exact match: exact match on 5 digits hides
partial learning, and at a floor that is precisely the signal we need.
"""
import sys
import time
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.arithmetic import PAD_IDX
from src.data.latent_rule import generate_dataset, LatentRuleDataset, RULE_NAMES
from src.models.dense import DenseModel

D_MODEL, N_LAYERS = 128, 2
TRAIN_N, EPOCHS, BS = 20000, 25, 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RUNGS = [('reverse', 0), ('sort', 1), ('prefixsum', 3), ('latent', None)]


def cfg():
    c = yaml.safe_load(open('configs/experiment.yaml'))
    c['model'].update(d_model=D_MODEL, n_layers=N_LAYERS,
                      max_seq_len=8, max_output_len=6)
    return c


def data_for(force_rule, seed=42):
    tr = generate_dataset(TRAIN_N, seed=seed, force_rule=force_rule)
    keys = {tuple(d) for d, _ in tr}
    te = generate_dataset(3000, seed=seed + 2, exclude=keys, force_rule=force_rule)
    return LatentRuleDataset(tr), LatentRuleDataset(te)


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
        ok = (p == y) | ~real
        exact += ok.all(dim=-1).sum().item()
        total += y.size(0)
    return exact / max(total, 1), tok_hit / max(tok_total, 1)


def run(name, force_rule, seed=1234):
    torch.manual_seed(seed)
    train, test = data_for(force_rule)
    tl = DataLoader(train, batch_size=BS, shuffle=True)
    testl = DataLoader(test, batch_size=256)

    model = DenseModel(cfg()).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=3e-5)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)['logits']
            loss = crit(out.reshape(-1, out.size(-1)), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    ex, tok = evaluate(model, testl)
    print(f'  {name:10s} exact {ex:.4f}   token {tok:.4f}   [{time.time()-t0:.0f}s]',
          flush=True)
    return ex, tok


def main():
    print('=== TASK LEARNABILITY PROBE (dense only) ===')
    print(f'device={DEVICE}  d={D_MODEL} L={N_LAYERS} {TRAIN_N} ex, {EPOCHS} ep')
    print('token accuracy is per-digit; exact is all 5 digits + EOS\n', flush=True)

    res = {}
    for name, fr in RUNGS:
        res[name] = run(name, fr)

    singles = [res[n][1] for n, fr in RUNGS if fr is not None]
    latent_tok = res['latent'][1]
    best_single = max(singles)

    print('\n=== READING ===')
    print(f'  best single-rule token acc : {best_single:.4f}')
    print(f'  latent-task token acc      : {latent_tok:.4f}')
    print(f'  chance (uniform digit)     : 0.1000')
    print()
    if best_single < 0.25:
        print('  >>> Even ONE algorithm is not learnable at this size. The bottleneck')
        print('      is the architecture, not the selector -- the output head pools the')
        print('      sequence, so order-dependent targets may not survive. Fix the task')
        print('      or the head before comparing anything.')
    elif latent_tok < best_single - 0.15:
        print('  >>> Single rules learn; the hidden selector is what hurts. That is the')
        print('      regime the coalition mechanism exists for -- run the comparison at')
        print('      this size.')
    else:
        print('  >>> Everything learns comfortably. Raise difficulty (longer sequences,')
        print('      more rules) until dense stops acing it, or the comparison will')
        print('      happen at a ceiling instead of a floor.')


if __name__ == '__main__':
    main()
