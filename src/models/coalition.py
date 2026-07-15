import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.shared import BaseModel


class CoalitionGraphFFN(nn.Module):
    def __init__(self, d_model: int, n_nodes: int, d_node: int, n_seeds: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.n_seeds = n_seeds

        self.nodes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_node),
                nn.GELU(),
                nn.Linear(d_node, d_model),
            )
            for _ in range(n_nodes)
        ])

        # Keys for seed selection — initialized with proper scale
        self.keys = nn.Parameter(torch.randn(n_nodes, d_model) / math.sqrt(d_model))

        # Edge weights for recruitment — init negative so most edges start weak
        # sigmoid(-2) ≈ 0.12, so initial recruitment signal is low
        self.edge_logits = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.5 - 2.0)

        # Recruitment threshold — start above typical recruit_signal so nodes must earn recruitment
        self.recruit_threshold = nn.Parameter(torch.tensor(0.5))

        self._temperature = 1.0

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, val):
        self._temperature = max(val, 0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch, seq_len, d_model = x.shape

        # Seed selection: use sequence-pooled representation to pick which nodes activate
        x_pooled = x.mean(dim=1)  # (batch, d_model)
        seed_scores = x_pooled @ self.keys.T  # (batch, n_nodes)

        # Top-k seed selection (always activates exactly n_seeds nodes)
        # Soft version: use softmax over seed scores for differentiable weighting
        tau = self._temperature

        # Hard top-k mask for seed activation
        _, top_indices = seed_scores.topk(self.n_seeds, dim=-1)
        hard_seed_mask = torch.zeros_like(seed_scores)
        hard_seed_mask.scatter_(1, top_indices, 1.0)

        # Soft seed scores (differentiable)
        soft_seed_scores = F.softmax(seed_scores / tau, dim=-1)

        # Straight-through: use hard mask in forward, soft scores in backward
        seed_activation = hard_seed_mask + soft_seed_scores - soft_seed_scores.detach()

        # Recruitment via learned edges
        edge_weights = torch.sigmoid(self.edge_logits)
        diag_mask = 1.0 - torch.eye(self.n_nodes, device=x.device)
        edge_weights = edge_weights * diag_mask

        # Recruitment signal: how strongly do active seeds recruit each node?
        recruit_signal = seed_activation @ edge_weights  # (batch, n_nodes)

        # Soft recruitment threshold
        recruit_activation = torch.sigmoid((recruit_signal - self.recruit_threshold) / tau)

        # Final activation: seed OR recruited (soft-OR)
        node_activation = seed_activation + recruit_activation - seed_activation * recruit_activation

        # Each node processes the full per-token input
        # Stack all node outputs: (batch, n_nodes, seq_len, d_model)
        node_outputs = torch.stack([node(x) for node in self.nodes], dim=1)

        # Weight by activation and sum across nodes
        # node_activation: (batch, n_nodes) -> (batch, n_nodes, 1, 1)
        weights = node_activation.unsqueeze(-1).unsqueeze(-1)
        weighted = (weights * node_outputs).sum(dim=1)  # (batch, seq_len, d_model)

        # Normalize by total activation
        activation_sum = node_activation.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1e-6)
        output = weighted / activation_sum  # (batch, seq_len, d_model)

        aux_data = {
            'node_activation': node_activation,
            'seed_activation': seed_activation,
            'recruit_activation': recruit_activation,
            'edge_weights': edge_weights.detach(),
            'seed_scores': seed_scores.detach(),
            'type': 'coalition',
        }
        return output, aux_data


class CoalitionModel(BaseModel):
    def __init__(self, config: dict):
        cc = config['coalition']

        def ffn_factory(d_model):
            return CoalitionGraphFFN(d_model, cc['n_nodes'], cc['d_node'], cc['n_seeds'])

        super().__init__(config, ffn_factory)
        self.coalition_config = cc

    def set_temperature(self, temperature: float):
        for layer in self.layers:
            if isinstance(layer.ffn, CoalitionGraphFFN):
                layer.ffn.temperature = temperature

    def get_temperature(self) -> float:
        for layer in self.layers:
            if isinstance(layer.ffn, CoalitionGraphFFN):
                return layer.ffn.temperature
        return 1.0

    def compute_temperature(self, epoch: int, total_epochs: int) -> float:
        cc = self.coalition_config
        warmup_end = int(total_epochs * cc['warmup_fraction'])
        if epoch < warmup_end:
            return cc['temp_start']
        progress = (epoch - warmup_end) / max(total_epochs - warmup_end, 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        temp = cc['temp_end'] + (cc['temp_start'] - cc['temp_end']) * cosine_decay
        return max(temp, cc['temp_end'])
