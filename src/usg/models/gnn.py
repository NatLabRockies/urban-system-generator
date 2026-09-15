"""Generalized masked neural network (G-NN) imputer.

Any subset of columns can be known. Hidden columns are replaced by per-column ``[MASK]`` embeddings and the
network predicts a categorical distribution for every imputable categorical column (and a Gaussian for
numeric ones). Trained with cross-entropy on hidden columns only, under randomly sampled masks.
"""

from __future__ import annotations

import torch
from torch import nn

from usg.models.layers import (
    CategoricalHeads,
    ColumnEmbedder,
    ResMLP,
    gumbel_argmax,
    masked_log_softmax,
)

ARCH_VERSION = 1

DEFAULT_ARCH = {
    "d_small": 16,
    "d_large": 32,
    "large_threshold": 16,
    "d_numeric": 16,
    "hidden": 512,
    "n_blocks": 4,
    "expansion": 2,
    "dropout": 0.1,
}


class GNNImputerNet(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        n_numeric: int,
        cat_head_mask: list[bool],
        num_head_mask: list[bool],
        arch: dict | None = None,
    ):
        super().__init__()
        a = {**DEFAULT_ARCH, **(arch or {})}
        self.arch = a
        self.n_numeric = n_numeric
        self.embed = ColumnEmbedder(
            cardinalities,
            n_numeric,
            d_small=a["d_small"],
            d_large=a["d_large"],
            large_threshold=a["large_threshold"],
            d_numeric=a["d_numeric"],
            n_flag_sets=1,
        )
        self.backbone = ResMLP(self.embed.out_dim, a["hidden"], a["n_blocks"], a["expansion"], a["dropout"])
        self.cat_heads = CategoricalHeads(a["hidden"], cardinalities, cat_head_mask)
        self.register_buffer("num_head_mask", torch.tensor(num_head_mask, dtype=torch.bool), persistent=False)
        self.num_head = nn.Linear(a["hidden"], 2 * n_numeric) if n_numeric else None

    def forward(
        self,
        cat: torch.Tensor,
        cat_known: torch.Tensor,
        num: torch.Tensor,
        num_known: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dtype = self.backbone.inp.weight.dtype
        flags = torch.cat([cat_known, num_known], dim=1).to(dtype)
        x = self.embed(cat, cat_known, num.to(dtype), num_known, [flags])
        h = self.backbone(x)
        out = {"cat_logits": self.cat_heads(h)}
        if self.num_head is not None:
            mean, logvar = self.num_head(h).chunk(2, dim=-1)
            out["num_mean"] = mean
            out["num_logvar"] = logvar.clamp(-10.0, 10.0)
        return out


@torch.no_grad()
def gnn_impute(
    net: GNNImputerNet,
    cat: torch.Tensor,
    cat_known: torch.Tensor,
    num: torch.Tensor,
    num_known: torch.Tensor,
    *,
    cat_imputable: torch.Tensor,
    num_imputable: torch.Tensor,
    mode: str = "argmax",
    temperature: torch.Tensor | None = None,
    rounds: int = 8,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Fill hidden imputable columns.

    Modes:
      * ``argmax``  — one pass; each hidden column takes its most likely class.
      * ``refine``  — MaskGIT-style: over ``rounds`` passes commit the most confident columns first and
        condition later passes on them (more coherent joint predictions).
      * ``sample``  — any-order ancestral sampling: random commit order, classes drawn from the predicted
        distribution (coherent joint samples for stock-level statistics).

    Returns ``cat_codes`` (B, Cc) with committed classes for hidden columns (known columns keep their code),
    ``cat_probs`` (B, Cc, K_max) the distribution each column was committed from, and numeric
    ``num_mean`` / ``num_std`` / ``num_value`` in transformed space.
    """
    if mode not in ("argmax", "refine", "sample"):
        raise ValueError(f"unknown G-NN decode mode {mode!r}")
    cat = cat.clone()
    cat_known = cat_known.clone()
    hidden = cat_imputable.unsqueeze(0) & ~cat_known
    probs_out = torch.zeros(
        (*cat.shape, net.cat_heads.index.shape[1]), dtype=torch.float32, device=cat.device
    )

    if mode == "argmax" or not bool(hidden.any()):
        out = net(cat, cat_known, num, num_known)
        logp = masked_log_softmax(out["cat_logits"], temperature)
        probs_out = logp.exp()
        cat = torch.where(hidden, logp.argmax(dim=-1), cat)
    else:
        total_hidden = hidden.sum(dim=1)
        remaining = hidden.clone()
        n_rounds = max(1, int(rounds))
        for r in range(n_rounds):
            if not bool(remaining.any()):
                break
            out = net(cat, cat_known, num, num_known)
            logp = masked_log_softmax(out["cat_logits"], temperature)
            if mode == "refine":
                score = logp.max(dim=-1).values
                choice = logp.argmax(dim=-1)
            else:
                score = torch.rand(remaining.shape, generator=generator, device=cat.device)
                choice = gumbel_argmax(logp, generator)
            score = score.masked_fill(~remaining, float("-inf"))
            rounds_left = n_rounds - r
            n_commit = torch.clamp(torch.ceil(total_hidden.float() / n_rounds), min=1).long()
            if rounds_left == 1:
                n_commit = remaining.sum(dim=1)
            order = score.argsort(dim=1, descending=True)
            ranks = torch.empty_like(order)
            ranks.scatter_(1, order, torch.arange(order.shape[1], device=cat.device).expand_as(order))
            commit = remaining & (ranks < n_commit.unsqueeze(1))
            cat = torch.where(commit, choice, cat)
            probs_out = torch.where(commit.unsqueeze(-1), logp.exp(), probs_out)
            cat_known = cat_known | commit
            remaining = remaining & ~commit
        out = net(cat, cat_known, num, num_known)
        # Known columns report their own one-hot-free distribution from the final pass.
        final_probs = masked_log_softmax(out["cat_logits"], temperature).exp()
        probs_out = torch.where(hidden.unsqueeze(-1), probs_out, final_probs)

    result = {"cat_codes": cat, "cat_probs": probs_out}
    if net.num_head is not None:
        mean = out["num_mean"]
        std = torch.exp(0.5 * out["num_logvar"])
        num_hidden = num_imputable.unsqueeze(0) & ~num_known
        if mode == "sample":
            noise = torch.randn(mean.shape, generator=generator, device=mean.device)
            value = mean + std * noise
        else:
            value = mean
        result["num_mean"] = mean
        result["num_std"] = std
        result["num_value"] = torch.where(num_hidden, value, num)
    return result


def parameter_count(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
