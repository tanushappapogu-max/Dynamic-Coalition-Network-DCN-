"""
Focused training script for the Coalition-Graph model only.
Logs coalition-specific diagnostics every epoch:
  - Per-node activation frequency (routing collapse?)
  - Average coalition size (too big? too small?)
  - Node position clustering (are nodes self-organizing?)
  - Temperature schedule
  - Gradient norms on routing params (keys, positions, recruit_threshold)
"""

import os
import sys
import yaml
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.arithmetic import create_datasets, PAD_IDX, decode_output, classify_expression
from src.models.coalition import CoalitionModel
from src.training.losses import coalition_load_balance_loss, coalition_size_loss, recruitment_encouragement_loss


def diagnose_coalition(model, loader, device, n_batches=5):
    model.eval()
    all_activations = []
    all_seed_acts = []
    all_recruit_acts = []

    with torch.no_grad():
        for i, (input_ids, _) in enumerate(loader):
            if i >= n_batches:
                break
            input_ids = input_ids.to(device)
            result = model(input_ids)
            if 'aux_data' in result:
                aux = result['aux_data'][0]
                all_activations.append(aux['node_activation'].cpu())
                all_seed_acts.append(aux['seed_activation'].cpu())
                all_recruit_acts.append(aux['recruit_activation'].cpu())

    if not all_activations:
        return {}

    activations = torch.cat(all_activations, dim=0)
    seed_acts = torch.cat(all_seed_acts, dim=0)
    recruit_acts = torch.cat(all_recruit_acts, dim=0)

    node_freq = activations.mean(dim=0)
    coalition_sizes = (activations > 0.5).float().sum(dim=1)

    diag = {
        'node_freq_mean': node_freq.mean().item(),
        'node_freq_std': node_freq.std().item(),
        'node_freq_min': node_freq.min().item(),
        'node_freq_max': node_freq.max().item(),
        'node_freqs': node_freq.tolist(),
        'coalition_size_mean': coalition_sizes.mean().item(),
        'coalition_size_std': coalition_sizes.std().item(),
        'coalition_size_min': coalition_sizes.min().item(),
        'coalition_size_max': coalition_sizes.max().item(),
        'seed_activation_mean': seed_acts.mean().item(),
        'recruit_activation_mean': recruit_acts.mean().item(),
        'temperature': model.get_temperature(),
    }

    # Position-based diagnostics
    for layer in model.layers:
        if hasattr(layer.ffn, 'positions'):
            pos = layer.ffn.positions.detach().cpu()
            pos_norm = pos / pos.norm(dim=1, keepdim=True)
            sim = (pos_norm @ pos_norm.T)
            mask = 1.0 - torch.eye(sim.shape[0])
            off_diag = sim[mask.bool()]
            diag['pos_sim_mean'] = off_diag.mean().item()
            diag['pos_sim_std'] = off_diag.std().item()
            diag['pos_sim_max'] = off_diag.max().item()
            diag['pos_sim_min'] = off_diag.min().item()
            n_close_pairs = (off_diag > 0.7).sum().item()
            diag['n_close_pairs'] = n_close_pairs
            break

    return diag


def get_grad_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None and ('positions' in name or 'keys' in name or 'recruit_threshold' in name):
            norms[name] = param.grad.norm().item()
    return norms


def train_one_epoch(model, loader, optimizer, criterion, config, device, grad_clip):
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_aux_loss = 0.0
    n_batches = 0
    grad_norms = {}

    cc = config['coalition']

    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        result = model(input_ids)
        logits = result['logits']

        task_loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

        aux_loss = torch.tensor(0.0, device=device)
        if 'aux_data' in result:
            for aux in result['aux_data']:
                if aux['type'] == 'coalition':
                    bal = coalition_load_balance_loss(aux['node_activation'])
                    sz = coalition_size_loss(aux['node_activation'], cc['target_coalition_size'])
                    rec = recruitment_encouragement_loss(aux['recruit_activation'])
                    aux_loss = aux_loss + cc['balance_coef'] * bal + cc['size_coef'] * sz + 0.01 * rec

        loss = task_loss + aux_loss

        optimizer.zero_grad()
        loss.backward()

        if n_batches == 0:
            grad_norms = get_grad_norms(model)

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_aux_loss += aux_loss.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'task_loss': total_task_loss / n_batches,
        'aux_loss': total_aux_loss / n_batches,
        'grad_norms': grad_norms,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        result = model(input_ids)
        logits = result['logits']
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=-1)
        correct += (preds == target_ids).all(dim=-1).sum().item()
        total += target_ids.size(0)

    return {
        'loss': total_loss / max(n_batches, 1),
        'accuracy': correct / max(total, 1),
    }


def main():
    with open('configs/experiment.yaml') as f:
        config = yaml.safe_load(f)

    small_mode = '--small' in sys.argv
    if small_mode:
        print('=== SMALL MODE: reduced dataset for quick testing ===')
        config['data']['train_size'] = 2000
        config['data']['val_size'] = 500
        config['data']['test_size'] = 500
        config['data']['gen_test_size'] = 200
        config['training']['epochs'] = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Data
    print('\nGenerating datasets...')
    datasets = create_datasets(config)
    for name, ds in datasets.items():
        print(f'  {name}: {len(ds)} examples')

    # Show samples
    print('\nSample expressions:')
    for i in range(8):
        expr, result = datasets['train'].data[i]
        print(f'  {expr} = {result}  [{classify_expression(expr)}]')

    bs = config['training']['batch_size']
    train_loader = DataLoader(datasets['train'], batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(datasets['val'], batch_size=bs, num_workers=0)
    test_loader = DataLoader(datasets['test'], batch_size=bs, num_workers=0)
    gen_loader = DataLoader(datasets['gen_test'], batch_size=bs, num_workers=0)

    # Model
    model = CoalitionModel(config).to(device)
    print(f'\nCoalition model: {model.count_parameters():,} parameters')

    tc = config['training']

    routing_params = model.get_routing_params()
    nonrouting_params = model.get_nonrouting_params()
    routing_lr = tc['learning_rate'] * 10

    print(f'Routing params: {sum(p.numel() for p in routing_params):,} (lr={routing_lr})')
    print(f'Other params:   {sum(p.numel() for p in nonrouting_params):,} (lr={tc["learning_rate"]})')

    optimizer = torch.optim.AdamW([
        {'params': nonrouting_params, 'lr': tc['learning_rate']},
        {'params': routing_params, 'lr': routing_lr},
    ], weight_decay=tc['weight_decay'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc['epochs'], eta_min=tc['learning_rate'] * 0.1
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    os.makedirs('results/coalition', exist_ok=True)
    log = []

    print(f'\nTraining for {tc["epochs"]} epochs...\n')
    print(f'{"Epoch":>5} {"Loss":>8} {"Task":>8} {"Aux":>8} {"ValLoss":>8} {"ValAcc":>8} {"Temp":>6} {"CoalSz":>7} {"NodeStd":>8} {"ClosePr":>8}')
    print('-' * 95)

    for epoch in range(tc['epochs']):
        start = time.time()

        temp = model.compute_temperature(epoch, tc['epochs'])
        model.set_temperature(temp)

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, config, device, tc['grad_clip'])
        val_metrics = evaluate(model, val_loader, criterion, device)
        diag = diagnose_coalition(model, val_loader, device)

        scheduler.step()
        elapsed = time.time() - start

        entry = {
            'epoch': epoch + 1,
            'elapsed': elapsed,
            **{f'train_{k}': v for k, v in train_metrics.items() if k != 'grad_norms'},
            **{f'val_{k}': v for k, v in val_metrics.items()},
            **diag,
            'grad_norms': train_metrics.get('grad_norms', {}),
        }
        log.append(entry)

        coal_sz = diag.get('coalition_size_mean', 0)
        node_std = diag.get('node_freq_std', 0)
        close_pr = diag.get('n_close_pairs', 0)

        print(
            f'{epoch+1:>5} '
            f'{train_metrics["loss"]:>8.4f} '
            f'{train_metrics["task_loss"]:>8.4f} '
            f'{train_metrics["aux_loss"]:>8.4f} '
            f'{val_metrics["loss"]:>8.4f} '
            f'{val_metrics["accuracy"]:>8.4f} '
            f'{temp:>6.3f} '
            f'{coal_sz:>7.1f} '
            f'{node_std:>8.4f} '
            f'{close_pr:>8}'
        )

        if (epoch + 1) % 5 == 0:
            print(f'  Node freqs: [{", ".join(f"{f:.2f}" for f in diag.get("node_freqs", []))}]')
            print(f'  Position sim: mean={diag.get("pos_sim_mean", 0):.3f} std={diag.get("pos_sim_std", 0):.3f} max={diag.get("pos_sim_max", 0):.3f}')
            if train_metrics.get('grad_norms'):
                for pname, norm in train_metrics['grad_norms'].items():
                    short = pname.split('.')[-1]
                    print(f'  Grad {short}: {norm:.6f}')
            print()

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'log': log,
            }, f'results/coalition/checkpoint_epoch{epoch+1}.pt')

    # Final evaluation
    print('\n' + '=' * 60)
    print('FINAL EVALUATION')
    print('=' * 60)

    test_metrics = evaluate(model, test_loader, criterion, device)
    gen_metrics = evaluate(model, gen_loader, criterion, device)
    final_diag = diagnose_coalition(model, test_loader, device, n_batches=20)

    print(f'  Test Accuracy (in-dist):     {test_metrics["accuracy"]:.4f}')
    print(f'  Test Accuracy (generalize):  {gen_metrics["accuracy"]:.4f}')
    print(f'  Avg Coalition Size:          {final_diag.get("coalition_size_mean", 0):.1f}')
    print(f'  Coalition Size Std:          {final_diag.get("coalition_size_std", 0):.2f}')
    print(f'  Node Freq Std:               {final_diag.get("node_freq_std", 0):.4f}')
    print(f'  Close Pairs (sim>0.7):       {final_diag.get("n_close_pairs", 0)}')
    print(f'  Position Sim Mean:           {final_diag.get("pos_sim_mean", 0):.4f}')
    print(f'  Recruit Mean:                {final_diag.get("recruit_activation_mean", 0):.4f}')

    print('\n--- HEALTH CHECK ---')
    coal_sz = final_diag.get('coalition_size_mean', 0)
    if coal_sz > 14:
        print(f'  WARNING: Coalition too large ({coal_sz:.1f}/16). Routing not selective.')
    elif coal_sz < 2:
        print(f'  WARNING: Coalition too small ({coal_sz:.1f}/16). Most nodes dead.')
    else:
        print(f'  OK: Coalition size {coal_sz:.1f}/16')

    node_std = final_diag.get('node_freq_std', 0)
    if node_std < 0.01:
        print(f'  WARNING: All nodes activate equally (std={node_std:.4f}). No specialization.')
    else:
        print(f'  OK: Node frequency std {node_std:.4f} — some differentiation')

    close_pairs = final_diag.get('n_close_pairs', 0)
    if close_pairs == 0:
        print(f'  WARNING: No close node pairs (sim>0.7). Nodes not clustering.')
    else:
        print(f'  OK: {close_pairs} close node pairs — clusters forming')

    recruit_mean = final_diag.get('recruit_activation_mean', 0)
    if recruit_mean < 0.01:
        print(f'  WARNING: Recruitment almost never fires (mean={recruit_mean:.4f})')
    elif recruit_mean > 0.8:
        print(f'  WARNING: Recruitment fires too often (mean={recruit_mean:.4f})')
    else:
        print(f'  OK: Recruitment activation mean {recruit_mean:.4f}')

    # Save everything
    torch.save({
        'model_state_dict': model.state_dict(),
        'log': log,
        'test_metrics': test_metrics,
        'gen_metrics': gen_metrics,
        'final_diagnostics': final_diag,
    }, 'results/coalition/final.pt')

    with open('results/coalition/training_log.json', 'w') as f:
        clean_log = [{k: v for k, v in entry.items() if k != 'grad_norms'} for entry in log]
        json.dump(clean_log, f, indent=2)

    print(f'\nResults saved to results/coalition/')


if __name__ == '__main__':
    main()
