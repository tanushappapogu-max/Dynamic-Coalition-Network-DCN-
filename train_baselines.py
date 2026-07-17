"""
Training script for Dense and MoE baseline models.
Same data, same epochs, same hyperparameters as coalition — fair fight.
"""

import os
import sys
import yaml
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.arithmetic import create_datasets, PAD_IDX
from src.models.dense import DenseModel
from src.models.moe import MoEModel
from src.training.losses import moe_load_balance_loss


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip, model_type, config):
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_aux_loss = 0.0
    n_batches = 0

    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        result = model(input_ids)
        logits = result['logits']

        task_loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

        aux_loss = torch.tensor(0.0, device=device)
        if model_type == 'moe' and 'aux_data' in result:
            mc = config['moe']
            for aux in result['aux_data']:
                if aux['type'] == 'moe':
                    bal = moe_load_balance_loss(
                        aux['gate_probs'], aux['top_k_indices'], mc['n_experts']
                    )
                    aux_loss = aux_loss + mc['balance_coef'] * bal

        loss = task_loss + aux_loss

        optimizer.zero_grad()
        loss.backward()
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
        match = (preds == target_ids) | (target_ids == PAD_IDX)
        correct += match.all(dim=-1).sum().item()
        total += target_ids.size(0)

    return {
        'loss': total_loss / max(n_batches, 1),
        'accuracy': correct / max(total, 1),
    }


@torch.no_grad()
def measure_inference_speed(model, loader, device, n_batches=50):
    model.eval()
    times = []
    for i, (input_ids, _) in enumerate(loader):
        if i >= n_batches:
            break
        input_ids = input_ids.to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(input_ids)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed / input_ids.size(0))
    return sum(times) / len(times) if times else 0.0


def train_model(model_type, config, device):
    print(f'\n{"="*60}')
    print(f'TRAINING {model_type.upper()} BASELINE')
    print(f'{"="*60}')

    datasets = create_datasets(config)

    bs = config['training']['batch_size']
    train_loader = DataLoader(datasets['train'], batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(datasets['val'], batch_size=bs, num_workers=0)
    test_loader = DataLoader(datasets['test'], batch_size=bs, num_workers=0)
    gen_loader = DataLoader(datasets['gen_test'], batch_size=bs, num_workers=0)

    if model_type == 'dense':
        model = DenseModel(config).to(device)
    else:
        model = MoEModel(config).to(device)

    print(f'{model_type.capitalize()} model: {model.count_parameters():,} parameters')

    tc = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc['learning_rate'], weight_decay=tc['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc['epochs'], eta_min=tc['learning_rate'] * 0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    out_dir = f'results/{model_type}'
    os.makedirs(out_dir, exist_ok=True)
    log = []

    print(f'\nTraining for {tc["epochs"]} epochs...\n')
    print(f'{"Epoch":>5} {"Loss":>8} {"Task":>8} {"Aux":>8} {"ValLoss":>8} {"ValAcc":>8}')
    print('-' * 55)

    for epoch in range(tc['epochs']):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, tc['grad_clip'], model_type, config)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start

        entry = {
            'epoch': epoch + 1,
            'elapsed': elapsed,
            'train_loss': train_metrics['loss'],
            'train_task_loss': train_metrics['task_loss'],
            'train_aux_loss': train_metrics['aux_loss'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
        }
        log.append(entry)

        print(
            f'{epoch+1:>5} '
            f'{train_metrics["loss"]:>8.4f} '
            f'{train_metrics["task_loss"]:>8.4f} '
            f'{train_metrics["aux_loss"]:>8.4f} '
            f'{val_metrics["loss"]:>8.4f} '
            f'{val_metrics["accuracy"]:>8.4f}'
        )

    print('\n' + '=' * 60)
    print('FINAL EVALUATION')
    print('=' * 60)

    test_metrics = evaluate(model, test_loader, criterion, device)
    gen_metrics = evaluate(model, gen_loader, criterion, device)
    inference_ms = measure_inference_speed(model, test_loader, device)

    print(f'  Test Accuracy (in-dist):     {test_metrics["accuracy"]:.4f}')
    print(f'  Test Accuracy (generalize):  {gen_metrics["accuracy"]:.4f}')
    print(f'  Inference Speed:             {inference_ms:.3f} ms/example')

    torch.save({
        'model_state_dict': model.state_dict(),
        'log': log,
        'test_metrics': test_metrics,
        'gen_metrics': gen_metrics,
        'inference_ms': inference_ms,
        'n_params': model.count_parameters(),
    }, f'{out_dir}/final.pt')

    with open(f'{out_dir}/training_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    print(f'\nResults saved to {out_dir}/')
    return {
        'test_accuracy': test_metrics['accuracy'],
        'gen_accuracy': gen_metrics['accuracy'],
        'inference_ms': inference_ms,
        'n_params': model.count_parameters(),
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

    results = {}

    flags = [a for a in sys.argv[1:] if a != '--small']
    run_all = len(flags) == 0

    if '--dense' in sys.argv or '--all' in sys.argv or run_all:
        results['dense'] = train_model('dense', config, device)

    if '--moe' in sys.argv or '--all' in sys.argv or run_all:
        results['moe'] = train_model('moe', config, device)

    if len(results) > 1:
        print('\n' + '=' * 60)
        print('COMPARISON')
        print('=' * 60)
        print(f'{"Model":<12} {"Params":>10} {"TestAcc":>10} {"GenAcc":>10} {"Speed(ms)":>10}')
        print('-' * 55)
        for name, r in results.items():
            print(f'{name:<12} {r["n_params"]:>10,} {r["test_accuracy"]:>10.4f} {r["gen_accuracy"]:>10.4f} {r["inference_ms"]:>10.3f}')


if __name__ == '__main__':
    main()
