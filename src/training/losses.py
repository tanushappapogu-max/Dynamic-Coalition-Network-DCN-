import torch
import torch.nn.functional as F


def moe_load_balance_loss(gate_probs: torch.Tensor, top_k_indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    n_tokens = gate_probs.shape[0]
    fraction_routed = torch.zeros(n_experts, device=gate_probs.device)
    for k in range(top_k_indices.shape[1]):
        counts = torch.bincount(top_k_indices[:, k], minlength=n_experts).float()
        fraction_routed += counts / n_tokens

    avg_gate_prob = gate_probs.mean(dim=0)
    return n_experts * (fraction_routed * avg_gate_prob).sum()


def coalition_load_balance_loss(node_activation: torch.Tensor) -> torch.Tensor:
    freq = node_activation.mean(dim=0)
    n_nodes = node_activation.shape[1]
    target = 1.0 / n_nodes
    return ((freq - target) ** 2).sum()


def coalition_size_loss(node_activation: torch.Tensor, target_size: float) -> torch.Tensor:
    avg_size = node_activation.sum(dim=1).mean()
    return (avg_size - target_size) ** 2


def recruitment_encouragement_loss(recruit_activation: torch.Tensor) -> torch.Tensor:
    avg_recruit = recruit_activation.mean()
    return torch.clamp(0.1 - avg_recruit, min=0.0) ** 2
