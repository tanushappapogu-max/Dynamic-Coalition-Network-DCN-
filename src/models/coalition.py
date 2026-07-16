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

        # Keys for seed selection
        self.keys = nn.Parameter(torch.randn(n_nodes, d_model) / math.sqrt(d_model))

        # Edge weights for recruitment — raw learned weights, NO sigmoid wrapper
        # Initialized so a few edges start strong enough to recruit
        self.edge_weights = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.3)

        # Bias per node for recruitment — start positive so recruitment is active early
        # The model learns to suppress recruitment where it's not useful
        self.recruit_bias = nn.Parameter(torch.ones(n_nodes) * 0.5)

        self._temperature = 1.0

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, val):
        self._temperature = max(val, 0.1)

    def _get_routing_params(self):
        return [self.keys, self.edge_weights, self.recruit_bias]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch, seq_len, d_model = x.shape
        tau = self._temperature

        # === SEED SELECTION ===
        # Pool sequence to get a routing vector
        x_pooled = x.mean(dim=1)  # (batch, d_model)

        # Score each node as a potential seed
        seed_scores = x_pooled @ self.keys.T  # (batch, n_nodes)

        # Soft seed weights via softmax (fully differentiable)
        soft_seed_weights = F.softmax(seed_scores / tau, dim=-1)  # (batch, n_nodes)

        # Hard top-k mask with straight-through gradient
        _, top_indices = seed_scores.topk(self.n_seeds, dim=-1)
        hard_seed_mask = torch.zeros_like(seed_scores)
        hard_seed_mask.scatter_(1, top_indices, 1.0)

        # Straight-through: hard forward, soft backward
        seed_activation = hard_seed_mask - soft_seed_weights.detach() + soft_seed_weights

        # === RECRUITMENT via attention-style scoring ===
        # Mask self-connections
        diag_mask = (1.0 - torch.eye(self.n_nodes, device=x.device))
        masked_edges = self.edge_weights * diag_mask  # (n_nodes, n_nodes)

        # Each seed broadcasts its recruitment score to neighbors
        # recruit_logits[b, j] = sum over seeds i: seed_activation[b,i] * edge_weights[i,j] + recruit_bias[j]
        recruit_logits = seed_activation @ masked_edges + self.recruit_bias  # (batch, n_nodes)

        # Recruitment uses a warmer temperature floor — aggressive annealing kills recruitment
        recruit_tau = max(tau, 0.5)
        recruit_activation = torch.sigmoid(recruit_logits / recruit_tau)  # (batch, n_nodes)

        # Zero out recruitment for nodes that are already seeds (they're already active)
        recruit_activation = recruit_activation * (1.0 - hard_seed_mask)

        # === COMBINE: seed + recruited ===
        # Seeds get weight from softmax, recruited get weight from sigmoid
        node_activation = seed_activation + recruit_activation  # (batch, n_nodes)

        # === COMPUTE ===
        node_outputs = torch.stack([node(x) for node in self.nodes], dim=1)  # (batch, n_nodes, seq_len, d_model)

        # Normalize by n_seeds (fixed) so recruited nodes ADD value without diluting seeds
        norm_weights = node_activation / self.n_seeds
        output = (norm_weights.unsqueeze(-1).unsqueeze(-1) * node_outputs).sum(dim=1)

        aux_data = {
            'node_activation': node_activation,
            'seed_activation': seed_activation,
            'recruit_activation': recruit_activation,
            'edge_weights': masked_edges.detach(),
            'recruit_logits': recruit_logits.detach(),
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

    def get_routing_params(self):
        """Return all routing-specific parameters for separate LR group."""
        routing_params = []
        for layer in self.layers:
            if isinstance(layer.ffn, CoalitionGraphFFN):
                routing_params.extend(layer.ffn._get_routing_params())
        return routing_params

    def get_nonrouting_params(self):
        """Return all non-routing parameters."""
        routing_ids = {id(p) for p in self.get_routing_params()}
        return [p for p in self.parameters() if id(p) not in routing_ids]

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
