"""Deep pattern probe for a trained coalition model.

Answers the questions that decide whether this is real self-organization:
  1. Do DIFFERENT node coalitions fire for different operation types?
     (addition vs multiplication vs parenthesized) — the core claim.
  2. Is each node specialized (fires selectively) or generic (fires always)?
  3. Does coalition size scale with input difficulty?
  4. Lesion test with MEANINGFUL accuracy: does settling compensate for damage?

Run AFTER train_coalition.py has saved results/coalition/final.pt.
Usage: python pattern_probe.py [--local]   (--local must match the training run's model size)
"""
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.arithmetic import create_datasets, PAD_IDX, classify_expression
from src.models.coalition import CoalitionModel


def main():
    with open('configs/experiment.yaml') as f:
        config = yaml.safe_load(f)

    # Must match whatever train run produced final.pt
    if '--local' in sys.argv:
        config['data']['train_size'] = 20000
        config['data']['val_size'] = 2000
        config['data']['test_size'] = 2000
        config['data']['gen_test_size'] = 1000
        config['model']['d_model'] = 128
        config['model']['n_layers'] = 2
        config['coalition']['d_node'] = 32

    device = torch.device('cpu')
    datasets = create_datasets(config)
    test_loader = DataLoader(datasets['test'], batch_size=128)

    model = CoalitionModel(config).to(device)
    ckpt = torch.load('results/coalition/final.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    # CRITICAL: temperature is not in state_dict — restore the annealed value the
    # model trained to, else everything runs soft/mushy at the default temp=1.0.
    model.set_temperature(config['coalition']['temp_end'])
    version = ckpt.get('version', config['coalition'].get('version', 7))
    n_nodes = config['coalition']['n_nodes']

    print(f'=== PATTERN PROBE (v{version}) ===')
    print(f'Model: {model.count_parameters():,} params')
    print(f'Test acc: {ckpt["test_metrics"]["accuracy"]:.4f}  '
          f'Hard: {ckpt.get("test_metrics_hard", {}).get("accuracy", float("nan")):.4f}')

    # === Collect per-type activation vectors (layer 0) ===
    type_acts = defaultdict(list)
    with torch.no_grad():
        for i in range(min(1500, len(datasets['test']))):
            input_ids, _ = datasets['test'][i]
            expr, _ = datasets['test'].data[i]
            out = model(input_ids.unsqueeze(0).to(device))
            act = out['aux_data'][0]['node_activation'][0].cpu().numpy()
            type_acts[classify_expression(expr)].append(act)

    types = sorted(type_acts.keys())
    print(f'\n=== COALITION SIGNATURE PER OPERATION TYPE (layer 0) ===')
    print('Mean activation of each node, grouped by expression type.')
    print('If self-organization is real, the rows should look DIFFERENT.\n')

    header = 'Type'.ljust(18) + ''.join(f'n{j:02d} ' for j in range(n_nodes))
    print(header)
    print('-' * len(header))
    means = {}
    for t in types:
        arr = np.array(type_acts[t])
        m = arr.mean(axis=0)
        means[t] = m
        row = t[:17].ljust(18) + ''.join(
            (f'\033[1m{v:.1f}\033[0m ' if v > 0.5 else f'{v:.1f} ') for v in m
        )
        print(row)

    # === How different ARE the coalitions across types? ===
    print(f'\n=== COALITION DIVERGENCE ===')
    print('Cosine similarity between type signatures (1.0 = identical routing).')
    print('Lower = types use different node sets = real specialization.\n')
    tlist = list(means.keys())
    print(' ' * 18 + ''.join(t[:8].ljust(9) for t in tlist))
    max_off = 0.0
    for a in tlist:
        row = a[:17].ljust(18)
        for b in tlist:
            va, vb = means[a], means[b]
            cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))
            row += f'{cos:.2f}'.ljust(9)
            if a != b:
                max_off = max(max_off, cos)
        print(row)

    # === Per-node selectivity: does each node PREFER a type? ===
    print(f'\n=== PER-NODE SPECIALIZATION ===')
    print(f'{"Node":>4}  {"overall":>8}  {"prefers":>16}  {"selectivity":>11}')
    print('-' * 48)
    n_specialized = 0
    for j in range(n_nodes):
        by_type = {t: means[t][j] for t in types}
        overall = np.mean([means[t][j] for t in types])
        pref = max(by_type, key=by_type.get)
        others = np.mean([v for t, v in by_type.items() if t != pref])
        selectivity = by_type[pref] - others
        if selectivity > 0.1:
            n_specialized += 1
        flag = ' *' if selectivity > 0.1 else ''
        print(f'{j:>4}  {overall:>8.3f}  {pref[:16]:>16}  {selectivity:>11.3f}{flag}')

    # === Coalition size vs difficulty ===
    print(f'\n=== COALITION SIZE BY DIFFICULTY ===')
    for t in types:
        arr = np.array(type_acts[t])
        sizes = (arr > 0.1).sum(axis=1)
        print(f'  {t:20s}: size {sizes.mean():.1f} +/- {sizes.std():.1f}')

    # === VERDICT ===
    print(f'\n=== VERDICT ===')
    print(f'  Max cross-type routing similarity: {max_off:.3f}  '
          f'({"TOO SIMILAR — same nodes for everything" if max_off > 0.98 else "types use distinguishable coalitions"})')
    print(f'  Specialized nodes: {n_specialized}/{n_nodes}  '
          f'({"good" if n_specialized >= 3 else "weak — most nodes are generic"})')

    les = ckpt.get('lesion_results', {})
    if les and 'settling_gain' in les:
        print(f'  Settling contribution: {les["settling_gain"]:+.4f}  '
              f'({"dynamics do real work" if les["settling_gain"] > 0.01 else "settling adds little"})')
        print(f'  Lesion busiest node: retains {les["retention"]:.0%} of healthy accuracy, '
              f're-settling recovers {les["resettle_gain"]:+.4f}')


if __name__ == '__main__':
    main()
