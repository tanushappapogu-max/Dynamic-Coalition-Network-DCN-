import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        out = self.token_emb(x) * math.sqrt(self.d_model) + self.pos_emb(positions)
        return self.dropout(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_module: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = ffn_module
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        if isinstance(ffn_out, tuple):
            ffn_out, aux_data = ffn_out
        else:
            aux_data = None
        x = self.norm2(x + self.dropout(ffn_out))
        if aux_data is not None:
            return x, aux_data
        return x


class OutputHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, max_output_len: int):
        super().__init__()
        self.max_output_len = max_output_len
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, max_output_len * vocab_size),
        )
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        logits = self.proj(x)
        return logits.view(-1, self.max_output_len, self.vocab_size)


class BaseModel(nn.Module):
    def __init__(self, config: dict, ffn_factory):
        super().__init__()
        mc = config['model']
        self.embedding = TokenEmbedding(mc['vocab_size'], mc['d_model'], mc['max_seq_len'], mc['dropout'])
        self.output_head = OutputHead(mc['d_model'], mc['vocab_size'], mc['max_output_len'])

        self.layers = nn.ModuleList()
        for _ in range(mc['n_layers']):
            ffn = ffn_factory(mc['d_model'])
            layer = TransformerEncoderLayer(mc['d_model'], mc['n_heads'], ffn, mc['dropout'])
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> dict:
        padding_mask = (x == 0)
        h = self.embedding(x)

        aux_data_list = []
        for layer in self.layers:
            out = layer(h, src_key_padding_mask=padding_mask)
            if isinstance(out, tuple):
                h, aux_data = out
                aux_data_list.append(aux_data)
            else:
                h = out

        logits = self.output_head(h, padding_mask)
        result = {'logits': logits}
        if aux_data_list:
            result['aux_data'] = aux_data_list
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
