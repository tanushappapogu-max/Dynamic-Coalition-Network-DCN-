import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.arithmetic import PAD_IDX, decode_output, classify_expression
from src.models.coalition import CoalitionModel


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = 0
    total = 0
    per_type = defaultdict(lambda: {'correct': 0, 'total': 0})

    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        result = model(input_ids)
        preds = result['logits'].argmax(dim=-1)
        matches = ((preds == target_ids) | (target_ids == PAD_IDX)).all(dim=-1)

        correct += matches.sum().item()
        total += target_ids.size(0)

    return {'accuracy': correct / max(total, 1), 'correct': correct, 'total': total}


@torch.no_grad()
def measure_inference_speed(model: nn.Module, loader: DataLoader, device: torch.device, n_samples: int = 1000) -> dict:
    model.eval()
    sample_batch = next(iter(loader))
    input_ids = sample_batch[0][:1].to(device)

    for _ in range(10):
        model(input_ids)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    count = 0
    for batch_input, _ in loader:
        for i in range(batch_input.size(0)):
            if count >= n_samples:
                break
            single = batch_input[i:i+1].to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(single)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            times.append(elapsed * 1000)
            count += 1
        if count >= n_samples:
            break

    return {
        'mean_ms': sum(times) / len(times),
        'median_ms': sorted(times)[len(times) // 2],
        'p95_ms': sorted(times)[int(len(times) * 0.95)],
        'n_samples': len(times),
    }


@torch.no_grad()
def compute_active_params(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if not isinstance(model, CoalitionModel):
        return {'total_params': total_params, 'active_ratio': 1.0, 'active_params_est': total_params, 'type': 'dense'}

    activation_sums = []
    n_nodes = None
    for batch_input, _ in loader:
        batch_input = batch_input.to(device)
        result = model(batch_input)
        if 'aux_data' in result:
            for aux in result['aux_data']:
                if aux['type'] == 'coalition':
                    act = aux['node_activation']
                    activation_sums.append(act.sum(dim=1).mean().item())
                    n_nodes = act.shape[1]
        break

    avg_active = sum(activation_sums) / max(len(activation_sums), 1)
    ratio = avg_active / n_nodes if n_nodes else 1.0
    return {
        'total_params': total_params,
        'active_ratio': ratio,
        'avg_coalition_size': avg_active,
        'n_nodes': n_nodes,
        'active_params_est': int(total_params * ratio),
        'type': 'coalition',
    }


@torch.no_grad()
def collect_activation_patterns(
    model: CoalitionModel, dataset, device: torch.device, n_samples: int = 1000
) -> list[dict]:
    model.eval()
    patterns = []

    for i in range(min(n_samples, len(dataset))):
        input_ids, target_ids = dataset[i]
        expr, result_str = dataset.data[i]

        input_batch = input_ids.unsqueeze(0).to(device)
        output = model(input_batch)

        layer_activations = []
        if 'aux_data' in output:
            for aux in output['aux_data']:
                if aux['type'] == 'coalition':
                    layer_activations.append(aux['node_activation'][0].cpu().numpy())

        patterns.append({
            'expression': expr,
            'result': result_str,
            'expr_type': classify_expression(expr),
            'activations': layer_activations,
        })

    return patterns
