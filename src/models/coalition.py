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

        self.keys = nn.Parameter(torch.randn(n_nodes, d_model) * 0.02)
        self.edge_logits = nn.Parameter(torch.zeros(n_nodes, n_nodes))
        self.seed_threshold = nn.Parameter(torch.tensor(0.0))
        self.recruit_threshold = nn.Parameter(torch.tensor(0.0))

        self._temperature = 1.0

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, val):
        self._temperature = max(val, 0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch, seq_len, d_model = x.shape
        x_pooled = x.mean(dim=1)

        seed_scores = x_pooled @ self.keys.T

        edge_weights = torch.sigmoid(self.edge_logits)
        mask = 1.0 - torch.eye(self.n_nodes, device=x.device)
        edge_weights = edge_weights * mask

        tau = self._temperature

        seed_activation = torch.sigmoid((seed_scores - self.seed_threshold) / tau)

        recruit_signal = seed_activation @ edge_weights

        recruit_activation = torch.sigmoid((recruit_signal - self.recruit_threshold) / tau)

        node_activation = seed_activation + recruit_activation - seed_activation * recruit_activation

        node_outputs = torch.stack([node(x_pooled) for node in self.nodes], dim=1)

        weighted = node_activation.unsqueeze(-1) * node_outputs
        output = weighted.sum(dim=1)
        activation_sum = node_activation.sum(dim=1, keepdim=True).clamp(min=1e-6)
        output = output / activation_sum

        output = output.unsqueeze(1).expand(-1, seq_len, -1)

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
