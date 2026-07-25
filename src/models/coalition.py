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

        # --- Original design: dense learned edge matrix ---
        # Each edge_logits[i,j] is a free parameter learned by gradient descent.
        # sigmoid(edge_logits) gives edge weights in [0,1].
        # Unlike position-based similarity, this can learn ANY graph structure:
        # complementary pairs, chains, hubs, cliques — whatever helps.
        # Initialized at 0 → sigmoid(0) = 0.5 → neutral prior, no edge preferred.
        self.edge_logits = nn.Parameter(torch.zeros(n_nodes, n_nodes))

        self.recruit_threshold = nn.Parameter(torch.tensor(0.0))

        self._temperature = 1.0
        self._recruit_enabled = True
        self._norm_mode = 'n_seeds'

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, val):
        self._temperature = max(val, 0.1)

    def _get_routing_params(self):
        return [self.keys, self.edge_logits, self.recruit_threshold]

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

        # === RECRUITMENT via learned edge matrix ===
        # edge_weights[i,j] = "how strongly node i recruits node j"
        # This is a free matrix — the network can learn ANY graph structure.
        # Self-connections masked so seeds don't recruit themselves.
        self_mask = torch.eye(self.n_nodes, device=x.device)
        edge_weights = torch.sigmoid(self.edge_logits) * (1.0 - self_mask)

        # recruit_signal[b,j] = sum_i seed_activation[b,i] * edge_weights[i,j]
        # "how much do all active seeds want node j?"
        recruit_signal = seed_activation @ edge_weights  # (batch, n_nodes)

        if self._recruit_enabled:
            recruit_tau = max(tau, 0.5)
            recruit_activation = torch.sigmoid((recruit_signal - self.recruit_threshold) / recruit_tau)
            # Seeds are already active — don't double-count
            recruit_activation = recruit_activation * (1.0 - hard_seed_mask)
        else:
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
            'recruit_signal': recruit_signal.detach(),
            'edge_weights': edge_weights.detach(),
            'seed_scores': seed_scores.detach(),
            'type': 'coalition',
        }
        return output, aux_data


class SubstitutiveCoalitionFFN(nn.Module):
    """v9: coalitions that SUBSTITUTE instead of accumulate.

    Two findings forced this design.

    1. No predictive gate. A node's drive is the energy of its own first-layer
       response -- it fires because the input actually excites its synapses.
       Nothing is trained to guess relevance before computing, so selection can
       depend on things a gate reading the raw input cannot see.

    2. Total activation mass is CONSERVED. v6-v8 all failed the same way, and
       the reason is mechanical: a node's gradient is dL/dout_i * w_i, so nodes
       that fire together with similar weights receive near-identical gradients
       and converge to the same function. Co-activation is an attractor toward
       redundancy. Every additive recruitment scheme therefore destroys the
       specialisation it needs -- measured as tsim rising with mass across
       v6/v7/v8, while the sparsest condition was both the most differentiated
       and the most accurate.

       Here activation is `budget * softmax(...)`, so the mass is exactly
       `budget` by construction. Lateral coupling can only REDISTRIBUTE it: a
       node joins the coalition by displacing another. Composition varies per
       input (which fixed top-k cannot express) while co-activation never grows.

    `set_substitutive(False)` swaps the softmax for a sigmoid, letting mass grow
    freely while holding the drive and coupling identical -- an exact isolation
    of substitution vs accumulation.
    """

    def __init__(self, d_model: int, n_nodes: int, d_node: int,
                 budget: float = 4.0, n_settle: int = 3,
                 budget_mode: str = 'fixed'):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.d_node = d_node
        self.budget = budget
        self.n_settle = n_settle
        # 'fixed' pins mass to `budget` for every input -- clean, but it throws
        # away variable coalition size, which was one of the original claims.
        # 'drive' sets the mass to the participation ratio of the drive vector,
        #     PR = (sum d)^2 / sum d^2
        # which is the effective number of nodes the input actually excites: n
        # when every node responds alike, 1 when a single node dominates. So
        # size is read off the response instead of being fixed or predicted by a
        # gate, and it starts near-dense and sparsifies only as drives separate.
        # 'uniform' switches selection off entirely: every node contributes with
        # weight 1/n. This is the control that separates two penalties that have
        # been confounded in every comparison so far -- the coalition FFN is 16
        # narrow blocks with a BLOCK-DIAGONAL second layer, which is strictly less
        # expressive than one wide dense FFN of equal width, quite apart from any
        # routing. Without this control, that structural tax gets misread as
        # evidence about routing.
        assert budget_mode in ('fixed', 'drive', 'uniform'), budget_mode
        self.budget_mode = budget_mode

        # Batched node weights: resonance reuses the real synapses, so the
        # thing that selects and the thing that computes are one object.
        s1 = 1.0 / math.sqrt(d_model)
        s2 = 1.0 / math.sqrt(d_node)
        self.W1 = nn.Parameter(torch.randn(n_nodes, d_model, d_node) * s1)
        self.b1 = nn.Parameter(torch.zeros(n_nodes, d_node))
        self.W2 = nn.Parameter(torch.randn(n_nodes, d_node, d_model) * s2)
        self.b2 = nn.Parameter(torch.zeros(n_nodes, d_model))

        # Who pulls whom into the coalition. Free to be positive (belongs with)
        # or negative (excludes). Cannot inflate mass -- only reallocate it.
        self.coupling = nn.Parameter(torch.zeros(n_nodes, n_nodes))
        # Only used in additive mode; init so starting mass ~= budget.
        self.threshold = nn.Parameter(torch.tensor(1.1))

        self._temperature = 1.0
        self._substitutive = True
        self._lesion = None

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, val):
        self._temperature = max(val, 0.1)

    def _get_routing_params(self):
        return [self.coupling, self.threshold]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch, seq_len, d_model = x.shape
        tau = self._temperature

        # === RESONANCE: every node listens with its own weights ===
        hidden = F.gelu(
            torch.einsum('bsd,ndh->bnsh', x, self.W1) + self.b1[None, :, None, :]
        )
        energy = hidden.pow(2).mean(dim=(2, 3)).sqrt()          # (batch, n_nodes)
        # Divisive normalisation: drive is relative to peers, mean ~1.
        drive = energy / (energy.mean(dim=1, keepdim=True) + 1e-8)

        if self._lesion is not None:
            drive = drive.clone()
            drive[:, self._lesion] = -1e4 if self._substitutive else 0.0

        C = self.coupling * (1.0 - torch.eye(self.n_nodes, device=x.device))

        if self.budget_mode == 'uniform':
            mass_budget = torch.full((batch, 1), float(self.n_nodes), device=x.device)
        elif self.budget_mode == 'drive':
            d = drive.clamp(min=1e-6)
            # Participation ratio: effective number of nodes the input excites.
            mass_budget = (d.sum(dim=1) ** 2 / d.pow(2).sum(dim=1)).unsqueeze(1)
        else:
            mass_budget = torch.full((batch, 1), self.budget, device=x.device)

        def activate(net):
            if self._substitutive:
                # Mass is exactly mass_budget: joining requires displacing.
                return mass_budget * F.softmax(net / tau, dim=1)
            # Ablation: mass free to grow, as in v6-v8.
            return torch.sigmoid((net - self.threshold) / tau)

        settle_delta = 0.0
        if self.budget_mode == 'uniform':
            # No selection at all -- pure structural control.
            a = torch.ones(batch, self.n_nodes, device=x.device)
            if self._lesion is not None:
                a = a.clone()
                a[:, self._lesion] = 0.0
        else:
            a = activate(drive - 1.0)
            for _ in range(self.n_settle):
                lateral = (a @ C) / self.n_nodes
                a_new = activate(drive - 1.0 + lateral)
                if self._lesion is not None:
                    a_new = a_new.clone()
                    a_new[:, self._lesion] = 0.0
                settle_delta = (a_new - a).abs().mean().item()
                a = a_new

        # === COMPUTE: weights sum to 1 so magnitude is controlled ===
        out_nodes = torch.einsum('bnsh,nhd->bnsd', hidden, self.W2) \
            + self.b2[None, :, None, :]
        w = a / a.sum(dim=1, keepdim=True).clamp(min=1e-8)
        output = (w[:, :, None, None] * out_nodes).sum(dim=1)

        aux_data = {
            'node_activation': a,
            'drive': drive.detach(),
            'coupling': C.detach(),
            'settle_delta': settle_delta,
            'mass': a.sum(dim=1).detach(),
            'type': 'coalition',
            'version': 9,
        }
        return output, aux_data


COALITION_FFN_TYPES = (CoalitionGraphFFN, SubstitutiveCoalitionFFN)


class CoalitionModel(BaseModel):
    def __init__(self, config: dict):
        cc = config['coalition']
        version = cc.get('version', 8)

        def ffn_factory(d_model):
            if version == 9:
                return SubstitutiveCoalitionFFN(
                    d_model, cc['n_nodes'], cc['d_node'],
                    budget=cc.get('budget', 4.0),
                    n_settle=cc.get('n_settle', 3),
                    budget_mode=cc.get('budget_mode', 'fixed'),
                )
            return CoalitionGraphFFN(
                d_model, cc['n_nodes'], cc['d_node'], cc['n_seeds'],
                d_pos=cc.get('d_pos', 32),
            )

        super().__init__(config, ffn_factory)
        self.coalition_config = cc
        self.version = version

    def _coalition_layers(self):
        return [l.ffn for l in self.layers if isinstance(l.ffn, COALITION_FFN_TYPES)]

    def get_routing_params(self):
        routing_params = []
        for ffn in self._coalition_layers():
            routing_params.extend(ffn._get_routing_params())
        return routing_params

    def get_nonrouting_params(self):
        routing_ids = {id(p) for p in self.get_routing_params()}
        return [p for p in self.parameters() if id(p) not in routing_ids]

    def set_temperature(self, temperature: float):
        for ffn in self._coalition_layers():
            ffn.temperature = temperature

    def set_recruitment(self, enabled: bool):
        """Ablation: turn proximity recruitment off to test whether it matters."""
        for ffn in self._coalition_layers():
            if hasattr(ffn, '_recruit_enabled'):
                ffn._recruit_enabled = enabled

    def set_norm_mode(self, mode: str):
        """'n_seeds' = v5 original; 'sum' = magnitude-controlled for fair ablation."""
        for ffn in self._coalition_layers():
            if hasattr(ffn, '_norm_mode'):
                ffn._norm_mode = mode

    def set_substitutive(self, enabled: bool):
        """v9: True = mass-conserving (join by displacing); False = mass free to grow."""
        for ffn in self._coalition_layers():
            if hasattr(ffn, '_substitutive'):
                ffn._substitutive = enabled

    def set_lesion(self, node_idx):
        """Knock out one node (None to clear) for the robustness test."""
        for ffn in self._coalition_layers():
            if hasattr(ffn, '_lesion'):
                ffn._lesion = node_idx

    def get_temperature(self) -> float:
        for ffn in self._coalition_layers():
            return ffn.temperature
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
