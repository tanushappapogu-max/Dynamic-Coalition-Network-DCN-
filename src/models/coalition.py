import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.shared import BaseModel


class CoalitionGraphFFN(nn.Module):
    def __init__(self, d_model: int, n_nodes: int, d_node: int, n_seeds: int, d_pos: int = 32):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.n_seeds = n_seeds
        self.d_pos = d_pos

        self.nodes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_node),
                nn.GELU(),
                nn.Linear(d_node, d_model),
            )
            for _ in range(n_nodes)
        ])

        # Keys for seed selection (input-dependent: "what task is this?")
        self.keys = nn.Parameter(torch.randn(n_nodes, d_model) / math.sqrt(d_model))

        # Node positions in learned embedding space (proximity = affinity)
        # Initialized on a unit sphere so distances are meaningful from the start
        pos_init = torch.randn(n_nodes, d_pos)
        pos_init = pos_init / pos_init.norm(dim=1, keepdim=True)
        self.positions = nn.Parameter(pos_init)

        self.recruit_threshold = nn.Parameter(torch.tensor(0.0))

        self._temperature = 1.0
        # Ablation switch: False => coalition is seeds ONLY (no proximity
        # recruitment). Isolates whether the recruitment mechanism does anything.
        self._recruit_enabled = True
        # Normalization: 'n_seeds' is v5's original (weights can sum to >1 when
        # recruits are added, so recruitment also inflates output magnitude).
        # 'sum' divides by the true activation sum, holding magnitude at 1 so an
        # ablation measures recruitment's CHOICE of nodes, not extra magnitude.
        self._norm_mode = 'n_seeds'

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, val):
        self._temperature = max(val, 0.1)

    def _get_routing_params(self):
        return [self.keys, self.positions, self.recruit_threshold]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch, seq_len, d_model = x.shape
        tau = self._temperature

        # === SEED SELECTION (input-dependent) ===
        x_pooled = x.mean(dim=1)  # (batch, d_model)
        seed_scores = x_pooled @ self.keys.T  # (batch, n_nodes)

        soft_seed_weights = F.softmax(seed_scores / tau, dim=-1)
        _, top_indices = seed_scores.topk(self.n_seeds, dim=-1)
        hard_seed_mask = torch.zeros_like(seed_scores)
        hard_seed_mask.scatter_(1, top_indices, 1.0)
        seed_activation = hard_seed_mask - soft_seed_weights.detach() + soft_seed_weights

        # === RECRUITMENT via position proximity ===
        # Compute pairwise similarity between all node positions
        pos_norm = F.normalize(self.positions, dim=1)  # (n_nodes, d_pos)
        similarity = pos_norm @ pos_norm.T  # (n_nodes, n_nodes), range [-1, 1]

        # For each batch element, compute how close each node is to the active seeds
        # seed_activation: (batch, n_nodes), similarity: (n_nodes, n_nodes)
        # proximity[b, j] = sum_i seed_activation[b, i] * similarity[i, j]
        proximity = seed_activation @ similarity  # (batch, n_nodes)

        # Subtract self-proximity (seeds shouldn't recruit themselves)
        proximity = proximity - seed_activation * 1.0  # remove self-similarity contribution

        if self._recruit_enabled:
            recruit_tau = max(tau, 0.5)
            recruit_activation = torch.sigmoid((proximity - self.recruit_threshold) / recruit_tau)
            recruit_activation = recruit_activation * (1.0 - hard_seed_mask)
        else:
            # Ablation: no neighbours pulled in — coalition is the seeds alone.
            recruit_activation = torch.zeros_like(seed_activation)

        # === COMBINE ===
        node_activation = seed_activation + recruit_activation

        # === COMPUTE ===
        node_outputs = torch.stack([node(x) for node in self.nodes], dim=1)

        if self._norm_mode == 'sum':
            denom = node_activation.sum(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            denom = self.n_seeds
        norm_weights = node_activation / denom
        output = (norm_weights.unsqueeze(-1).unsqueeze(-1) * node_outputs).sum(dim=1)

        aux_data = {
            'node_activation': node_activation,
            'seed_activation': seed_activation,
            'recruit_activation': recruit_activation,
            'proximity': proximity.detach(),
            'similarity_matrix': similarity.detach(),
            'positions': self.positions.detach(),
            'seed_scores': seed_scores.detach(),
            'type': 'coalition',
        }
        return output, aux_data


class CoalitionModel(BaseModel):
    def __init__(self, config: dict):
        cc = config['coalition']

        def ffn_factory(d_model):
            return CoalitionGraphFFN(
                d_model, cc['n_nodes'], cc['d_node'], cc['n_seeds'],
                d_pos=cc.get('d_pos', 32),
            )

        super().__init__(config, ffn_factory)
        self.coalition_config = cc

    def get_routing_params(self):
        routing_params = []
        for layer in self.layers:
            if isinstance(layer.ffn, CoalitionGraphFFN):
                routing_params.extend(layer.ffn._get_routing_params())
        return routing_params

    def get_nonrouting_params(self):
        routing_ids = {id(p) for p in self.get_routing_params()}
        return [p for p in self.parameters() if id(p) not in routing_ids]

    def set_temperature(self, temperature: float):
        for layer in self.layers:
            if isinstance(layer.ffn, CoalitionGraphFFN):
                layer.ffn.temperature = temperature

    def set_recruitment(self, enabled: bool):
        """Ablation: turn proximity recruitment off to test whether it matters."""
        for layer in self.layers:
            if isinstance(layer.ffn, CoalitionGraphFFN):
                layer.ffn._recruit_enabled = enabled

    def set_norm_mode(self, mode: str):
        """'n_seeds' = v5 original; 'sum' = magnitude-controlled for fair ablation."""
        for layer in self.layers:
            if isinstance(layer.ffn, CoalitionGraphFFN):
                layer.ffn._norm_mode = mode

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
