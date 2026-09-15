"""Building blocks shared by the G-NN imputer and the TabDDPM denoiser (pure torch)."""

from __future__ import annotations

import itertools
import math

import torch
from torch import nn

PAD_LOGIT = -1e9  # logit for padded (non-existent) classes; finite so log-space math never produces NaN


class ColumnEmbedder(nn.Module):
    """Embeds every column and concatenates the result.

    Categorical column ``c`` with ``K`` classes owns ``K + 1`` embedding rows; row ``K`` is its learned
    ``[MASK]`` vector. Columns with fewer than ``large_threshold`` classes use ``d_small`` dimensions,
    the rest ``d_large``. Numeric columns use a per-column affine embedding with a learned mask vector.
    ``flags`` (float tensors of shape ``(B, Cc + Cn)``) are appended verbatim, e.g. the known-indicator.
    """

    def __init__(
        self,
        cardinalities: list[int],
        n_numeric: int,
        *,
        d_small: int = 16,
        d_large: int = 32,
        large_threshold: int = 16,
        d_numeric: int = 16,
        n_flag_sets: int = 1,
    ):
        super().__init__()
        self.n_cat = len(cardinalities)
        self.n_num = n_numeric
        self.register_buffer("mask_index", torch.tensor(cardinalities, dtype=torch.long), persistent=False)
        self._groups: list[tuple[str, int, int]] = []
        splits = (
            ("small", d_small, [i for i, k in enumerate(cardinalities) if k < large_threshold]),
            ("large", d_large, [i for i, k in enumerate(cardinalities) if k >= large_threshold]),
        )
        for name, dim, cols in splits:
            if not cols:
                continue
            sizes = [cardinalities[i] + 1 for i in cols]
            offsets = [0, *itertools.accumulate(sizes)][:-1]
            self.register_buffer(f"{name}_cols", torch.tensor(cols, dtype=torch.long), persistent=False)
            self.register_buffer(f"{name}_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
            table = nn.Embedding(sum(sizes), dim)
            nn.init.normal_(table.weight, std=1.0 / math.sqrt(dim))
            self.add_module(f"{name}_table", table)
            self._groups.append((name, len(cols), dim))
        if n_numeric:
            self.num_weight = nn.Parameter(torch.randn(n_numeric, d_numeric) / math.sqrt(d_numeric))
            self.num_bias = nn.Parameter(torch.zeros(n_numeric, d_numeric))
            self.num_mask = nn.Parameter(torch.randn(n_numeric, d_numeric) * 0.02)
        self.out_dim = (
            sum(n * d for _, n, d in self._groups) + n_numeric * d_numeric + n_flag_sets * (self.n_cat + n_numeric)
        )

    def forward(
        self,
        cat_codes: torch.Tensor,
        cat_use_value: torch.Tensor,
        num_values: torch.Tensor,
        num_use_value: torch.Tensor,
        flags: list[torch.Tensor] = (),
    ) -> torch.Tensor:
        codes = torch.where(cat_use_value, cat_codes.clamp(min=0), self.mask_index.expand_as(cat_codes))
        parts = []
        for name, _, _ in self._groups:
            cols = getattr(self, f"{name}_cols")
            offsets = getattr(self, f"{name}_offsets")
            emb = getattr(self, f"{name}_table")(codes[:, cols] + offsets)
            parts.append(emb.flatten(1))
        if self.n_num:
            v = num_values.unsqueeze(-1) * self.num_weight + self.num_bias
            v = torch.where(num_use_value.unsqueeze(-1), v, self.num_mask.expand_as(v))
            parts.append(v.flatten(1))
        parts.extend(flags)
        return torch.cat(parts, dim=-1)


class ResBlock(nn.Module):
    """Pre-LayerNorm residual MLP block with optional FiLM conditioning (used for diffusion timesteps)."""

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0, cond_dim: int | None = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.film = None
        if cond_dim:
            self.film = nn.Linear(cond_dim, 2 * dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm(h)
        if self.film is not None:
            scale, shift = self.film(cond).chunk(2, dim=-1)
            x = x * (1 + scale) + shift
        return h + self.fc2(self.drop(self.act(self.fc1(x))))


class ResMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        n_blocks: int,
        expansion: int = 2,
        dropout: float = 0.0,
        cond_dim: int | None = None,
    ):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList(ResBlock(hidden, expansion, dropout, cond_dim) for _ in range(n_blocks))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.inp(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.norm(h)


class CategoricalHeads(nn.Module):
    """One projection for all class logits, gathered into a padded ``(B, Cc, K_max)`` tensor.

    Only columns in ``head_mask`` get real logits; others (conditioning inputs) are entirely padding.
    """

    def __init__(self, hidden: int, cardinalities: list[int], head_mask: list[bool]):
        super().__init__()
        k_max = max(cardinalities, default=1)
        head_cols = [i for i, (k, m) in enumerate(zip(cardinalities, head_mask)) if m and k > 0]
        total = sum(cardinalities[i] for i in head_cols)
        index = torch.full((len(cardinalities), k_max), total, dtype=torch.long)
        offset = 0
        for i in head_cols:
            k = cardinalities[i]
            index[i, :k] = torch.arange(offset, offset + k)
            offset += k
        valid = torch.zeros((len(cardinalities), k_max), dtype=torch.bool)
        for i in head_cols:
            valid[i, : cardinalities[i]] = True
        self.register_buffer("index", index, persistent=False)
        self.register_buffer("valid", valid, persistent=False)
        self.proj = nn.Linear(hidden, total) if total else None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        pad = h.new_full((h.shape[0], 1), PAD_LOGIT)
        flat = pad if self.proj is None else torch.cat([self.proj(h), pad], dim=1)
        return flat[:, self.index]


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps ``t`` (shape ``(B,)``)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def masked_log_softmax(logits: torch.Tensor, temperature: torch.Tensor | None = None) -> torch.Tensor:
    if temperature is not None:
        logits = logits / temperature.view(1, -1, 1)
    return torch.log_softmax(logits.float(), dim=-1)


def gumbel_argmax(log_probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample one class per ``(B, C)`` from log-probabilities over the last dim (device-independent)."""
    u = torch.rand(log_probs.shape, generator=generator, device=log_probs.device, dtype=torch.float32)
    gumbel = -torch.log(-torch.log(u.clamp_(1e-10, 1.0 - 1e-10)))
    return (log_probs + gumbel).argmax(dim=-1)
