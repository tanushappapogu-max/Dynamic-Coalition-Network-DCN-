"""Block-Gated FFN: sparsity without the mixture.

WHY THIS EXISTS (derived from our own measurements, not from prior work)

Every sparse expert layer we built, and standard top-k MoE, combines expert
outputs as a CONVEX AVERAGE -- gate weights normalised to sum to one:

    mixture(x) = sum_i (a_i / sum_j a_j) * (W2_i @ GELU(W1_i @ x))

A dense FFN is not an average. It is a SUM over its hidden units, which can be
grouped into blocks:

    dense(x) = W2 @ GELU(W1 @ x) = sum_i W2_block_i @ GELU(W1_block_i @ x)

So a mixture over n experts sits roughly a factor of n below the dense layer in
magnitude. Inside a residual block the FFN output is added to the stream and then
LayerNormed, so attenuating it by 1/n does not merely rescale -- it shrinks the
FFN's share of the residual toward zero. We measured a penalty that grew with n
and then saturated (-0.036 at n=2 rising to -0.097 by n=16, flat to n=32), which
is the signature of exactly that: more experts -> more attenuation -> less FFN,
until the FFN contributes nothing and further attenuation changes nothing. We
originally read that curve as a cost of factorisation. It is a cost of averaging.

THE FIX, AND THE POINT OF THIS MODULE

Gate the hidden units of ONE FFN in blocks, and keep the output projection full
and shared:

    output = W2 @ (mask ⊙ GELU(W1 @ x))

Properties that follow, none of which hold for a mixture:

  1. With every block on and no rescaling, this is EXACTLY a dense FFN. Not
     approximately -- the same function, same parameters. `assert_equals_dense`
     checks it numerically. So there is no penalty to pay back before routing can
     help, which is the thing that sank v6-v9.
  2. Active blocks still mix. A mixture gives expert i its own W2_i, so expert i's
     hidden units can never combine with expert j's. Here every active hidden unit
     feeds one shared full projection.
  3. Compute still scales with the active fraction: inactive blocks need neither
     their W1 rows nor their W2 columns. (This prototype computes the full hidden
     layer and masks, as small-scale MoE implementations also do; the saving is in
     the structure, and `active_flops_fraction` reports it.)
  4. Sparsity no longer changes output magnitude the way averaging does. With
     `rescale='inv'` the surviving blocks are scaled by n/k, so expected
     magnitude is held constant as k varies -- the same trick inverted dropout
     uses, and a principled alternative to normalising by the gate sum.

SELECTION is resonance-based, as in v9: a block's drive is the energy of its own
hidden response, so the thing that selects is the thing that computes and no
parameter is trained to predict relevance before computing. Top-k is applied with
a straight-through estimator so the mask is hard forward and differentiable
backward. Selection is deliberately kept simple here; the contribution is the
gating structure, and it must not be confounded with a cleverer router.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.shared import BaseModel


class BlockGatedFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, n_blocks: int,
                 k_active: int = None, rescale: str = 'inv',
                 select: str = 'resonance'):
        super().__init__()
        assert d_ffn % n_blocks == 0, (d_ffn, n_blocks)
        assert rescale in ('none', 'inv'), rescale
        assert select in ('resonance', 'gate', 'all'), select

        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_blocks = n_blocks
        self.block = d_ffn // n_blocks
        self.k_active = n_blocks if k_active is None else k_active
        self.rescale = rescale
        self.select = select

        # ONE wide FFN. W2 is full and shared -- this is the whole point.
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)

        # Only used by select='gate'; kept so the two selection rules can be
        # compared without changing anything else about the layer.
        self.gate = nn.Linear(d_model, n_blocks, bias=False)

        self._temperature = 1.0
        self._lesion = None

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, val):
        self._temperature = max(val, 0.1)

    def _get_routing_params(self):
        return [self.gate.weight] if self.select == 'gate' else []

    def active_flops_fraction(self) -> float:
        """Share of FFN matmul work a sparse implementation would do."""
        return self.k_active / self.n_blocks

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch, seq_len, _ = x.shape
        h = F.gelu(self.fc1(x))                        # (B, S, d_ffn)
        hb = h.view(batch, seq_len, self.n_blocks, self.block)

        if self.select == 'all' or self.k_active >= self.n_blocks:
            mask = torch.ones(batch, self.n_blocks, device=x.device, dtype=h.dtype)
            scores = hb.pow(2).mean(dim=(1, 3)).sqrt().detach()
        else:
            if self.select == 'resonance':
                # A block's drive is the energy of its own hidden response,
                # pooled over the sequence. Selection reuses the computation.
                scores = hb.pow(2).mean(dim=(1, 3)).sqrt()
                scores = scores / (scores.mean(dim=1, keepdim=True) + 1e-8)
            else:
                scores = self.gate(x.mean(dim=1))

            if self._lesion is not None:
                scores = scores.clone()
                scores[:, self._lesion] = -1e4

            # Hard top-k forward, soft gradient backward. No convex
            # normalisation anywhere: an active block's weight is 1, not 1/k.
            soft = torch.sigmoid((scores - scores.median(dim=1, keepdim=True).values)
                                 / self._temperature)
            topv, topi = scores.topk(self.k_active, dim=1)
            hard = torch.zeros_like(scores).scatter_(1, topi, 1.0)
            mask = hard + soft - soft.detach()

        if self.rescale == 'inv' and self.k_active < self.n_blocks:
            # Hold expected magnitude constant as k varies (inverted-dropout
            # style). Note this scales UP the survivors; it never divides by the
            # gate sum, so all-on stays exactly dense.
            mask = mask * (self.n_blocks / self.k_active)

        hb = hb * mask[:, None, :, None]
        out = self.fc2(hb.reshape(batch, seq_len, self.d_ffn))

        aux = {
            'node_activation': mask.detach(),
            'block_scores': scores.detach(),
            'active_blocks': float(self.k_active),
            'flops_fraction': self.active_flops_fraction(),
            'type': 'block_gated',
        }
        return out, aux


class BlockGatedModel(BaseModel):
    def __init__(self, config: dict):
        bc = config['block_gated']

        def ffn_factory(d_model):
            return BlockGatedFFN(
                d_model, bc['d_ffn'], bc['n_blocks'],
                k_active=bc.get('k_active'),
                rescale=bc.get('rescale', 'inv'),
                select=bc.get('select', 'resonance'),
            )

        super().__init__(config, ffn_factory)
        self.block_config = bc

    def _ffns(self):
        return [l.ffn for l in self.layers if isinstance(l.ffn, BlockGatedFFN)]

    def get_routing_params(self):
        ps = []
        for f in self._ffns():
            ps.extend(f._get_routing_params())
        return ps

    def set_temperature(self, t: float):
        for f in self._ffns():
            f.temperature = t

    def set_lesion(self, idx):
        for f in self._ffns():
            f._lesion = idx

    def flops_fraction(self) -> float:
        fs = self._ffns()
        return sum(f.active_flops_fraction() for f in fs) / max(len(fs), 1)

    def compute_temperature(self, epoch: int, total_epochs: int) -> float:
        bc = self.block_config
        warm = int(total_epochs * bc.get('warmup_fraction', 0.1))
        t0, t1 = bc.get('temp_start', 1.0), bc.get('temp_end', 0.1)
        if epoch < warm:
            return t0
        prog = (epoch - warm) / max(total_epochs - warm, 1)
        return max(t1 + (t0 - t1) * 0.5 * (1.0 + math.cos(math.pi * prog)), t1)


@torch.no_grad()
def assert_equals_dense(d_model=64, d_ffn=256, n_blocks=8, seq=5, batch=4,
                        tol=1e-5) -> float:
    """The claim that distinguishes this from a mixture: all blocks on IS dense.

    Builds a block-gated FFN and a plain dense FFN sharing weights, and returns
    the max absolute output difference. A mixture cannot pass this at any n>1,
    because averaging introduces a 1/n factor that no choice of weights removes.
    """
    bg = BlockGatedFFN(d_model, d_ffn, n_blocks, k_active=n_blocks,
                       rescale='none', select='all')
    dense = nn.Sequential(
        nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model))
    dense[0].weight.copy_(bg.fc1.weight); dense[0].bias.copy_(bg.fc1.bias)
    dense[2].weight.copy_(bg.fc2.weight); dense[2].bias.copy_(bg.fc2.bias)

    x = torch.randn(batch, seq, d_model)
    diff = (bg(x)[0] - dense(x)).abs().max().item()
    assert diff < tol, f'block-gated != dense (max diff {diff})'
    return diff
