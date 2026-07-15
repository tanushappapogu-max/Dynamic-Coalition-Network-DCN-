import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.shared import BaseModel


class MoEFFN(nn.Module):
    def __init__(self, d_model: int, n_experts: int, d_expert: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_expert),
                nn.GELU(),
                nn.Linear(d_expert, d_model),
            )
            for _ in range(n_experts)
        ])
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)

        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)

        top_k_probs, top_k_indices = gate_probs.topk(self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]
            weight = top_k_probs[:, k].unsqueeze(-1)
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weight[mask] * expert_output

        output = output.view(batch, seq_len, d_model)

        aux_data = {
            'gate_probs': gate_probs,
            'top_k_indices': top_k_indices,
            'type': 'moe',
        }
        return output, aux_data


class MoEModel(BaseModel):
    def __init__(self, config: dict):
        mc = config['moe']

        def ffn_factory(d_model):
            return MoEFFN(d_model, mc['n_experts'], mc['d_expert'], mc['top_k'])

        super().__init__(config, ffn_factory)
