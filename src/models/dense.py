import torch
import torch.nn as nn

from src.models.shared import BaseModel


class DenseFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseModel(BaseModel):
    def __init__(self, config: dict):
        d_ffn = config['dense']['d_ffn']

        def ffn_factory(d_model):
            return DenseFFN(d_model, d_ffn)

        super().__init__(config, ffn_factory)
