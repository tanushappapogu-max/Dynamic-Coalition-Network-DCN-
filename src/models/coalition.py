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

        # Per-token attention over coalition members
        self.token_gate = nn.Linear(d_model, n_nodes, bias=False)
        nn.init.normal_(self.token_gate.weight, std=0.02)

        self._temperature = 1.0

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

        recruit_tau = max(tau, 0.5)
        recruit_activation = torch.sigmoid((proximity - self.recruit_threshold) / recruit_tau)
        recruit_activation = recruit_activation * (1.0 - hard_seed_mask)

        # === COMBINE ===
        node_activation = seed_activation + recruit_activation

        # === COMPUTE ===
        node_outputs = torch.stack([node(x) for node in self.nodes], dim=1)
        # node_outputs: (batch, n_nodes, seq_len, d_model)

        # === PER-TOKEN WEIGHTING ===
        token_logits = self.token_gate(x)  # (batch, seq_len, n_nodes)

        # Fold routing strength into softmax via log(activation)
        # At init (token_logits ≈ 0): softmax(log(act)) ∝ act → recovers old behavior exactly
        log_act = torch.log(node_activation.clamp(min=1e-8)).unsqueeze(1)
        combined_logits = token_logits + log_act

        active_mask = (node_activation.detach() > 0.01).unsqueeze(1)
        combined_logits = combined_logits.masked_fill(~active_mask, -1e9)
        token_weights = F.softmax(combined_logits, dim=-1)

        # Scale to match original output magnitude: sum(activation) / n_seeds
        scale = node_activation.sum(dim=-1, keepdim=True).unsqueeze(1) / self.n_seeds
        final_weights = token_weights * scale

        output = torch.einsum('bsn,bnsd->bsd', final_weights, node_outputs)

        aux_data = {
            'node_activation': node_activation,
            'seed_activation': seed_activation,
            'recruit_activation': recruit_activation,
            'proximity': proximity.detach(),
            'similarity_matrix': similarity.detach(),
            'positions': self.positions.detach(),
            'seed_scores': seed_scores.detach(),
            'token_attn': token_weights.detach(),
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
