"""Mask-conditional TabDDPM: denoiser network and conditional sampler.

Known columns (and conditioning inputs) stay clean; only hidden imputable columns are diffused. The denoiser
sees clean known values, noisy hidden values, a clean-known flag per column and the timestep (FiLM), and
predicts ``p(x_0)`` for categorical columns and ``eps`` for numeric ones.
"""

from __future__ import annotations

import itertools

import torch
from torch import nn

from usg.models.diffusion import (
    DiffusionSchedule,
    cat_posterior,
    num_posterior,
    num_predict_x0,
    uniform_log_probs,
)
from usg.models.layers import (
    CategoricalHeads,
    ColumnEmbedder,
    ResMLP,
    gumbel_argmax,
    masked_log_softmax,
    timestep_embedding,
)

ARCH_VERSION = 1

DEFAULT_ARCH = {
    "d_small": 16,
    "d_large": 32,
    "large_threshold": 16,
    "d_numeric": 16,
    "hidden": 768,
    "n_blocks": 4,
    "expansion": 2,
    "dropout": 0.1,
    "t_embed": 128,
}

DEFAULT_DIFFUSION = {"timesteps": 1000, "schedule": "cosine"}


class TabDDPMDenoiser(nn.Module):
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
        hidden = a["hidden"]
        self.t_mlp = nn.Sequential(nn.Linear(a["t_embed"], hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.backbone = ResMLP(self.embed.out_dim, hidden, a["n_blocks"], a["expansion"], a["dropout"], cond_dim=hidden)
        self.cat_heads = CategoricalHeads(hidden, cardinalities, cat_head_mask)
        self.eps_head = nn.Linear(hidden, n_numeric) if n_numeric else None
        self.register_buffer("card", torch.tensor([max(k, 1) for k in cardinalities], dtype=torch.float32), persistent=False)

    def forward(
        self,
        cat_t: torch.Tensor,
        cat_use_value: torch.Tensor,
        num_t: torch.Tensor,
        num_use_value: torch.Tensor,
        clean_known: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dtype = self.backbone.inp.weight.dtype
        cond = self.t_mlp(timestep_embedding(t, self.arch["t_embed"]).to(dtype))
        x = self.embed(cat_t, cat_use_value, num_t.to(dtype), num_use_value, [clean_known.to(dtype)])
        h = self.backbone(x, cond)
        out = {"x0_logits": self.cat_heads(h)}
        if self.eps_head is not None:
            out["eps"] = self.eps_head(h)
        return out


@torch.no_grad()
def tabddpm_sample(
    net: TabDDPMDenoiser,
    schedule: DiffusionSchedule,
    cat: torch.Tensor,
    cat_known: torch.Tensor,
    num: torch.Tensor,
    num_known: torch.Tensor,
    *,
    cat_imputable: torch.Tensor,
    num_imputable: torch.Tensor,
    n_samples: int = 16,
    steps: int = 100,
    generator: torch.Generator | None = None,
    temperature: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Draw ``n_samples`` completions per row. Returns ``cat`` (B, K, Cc) codes and ``num`` (B, K, Cn)."""
    B, Cc = cat.shape
    K = int(n_samples)
    device = cat.device
    valid = net.cat_heads.valid
    card = net.card

    cat_r = cat.repeat_interleave(K, dim=0)
    known_r = cat_known.repeat_interleave(K, dim=0)
    num_r = num.repeat_interleave(K, dim=0)
    num_known_r = num_known.repeat_interleave(K, dim=0)
    hidden_cat = cat_imputable.unsqueeze(0) & ~known_r
    hidden_num = num_imputable.unsqueeze(0) & ~num_known_r
    clean_known = torch.cat([known_r, num_known_r], dim=1)

    init = gumbel_argmax(uniform_log_probs(valid, card).expand(B * K, -1, -1), generator)
    cat_t = torch.where(hidden_cat, init, cat_r)
    noise = torch.randn(num_r.shape, generator=generator, device=device)
    num_t = torch.where(hidden_num, noise, num_r)
    cat_use = hidden_cat | known_r
    num_use = hidden_num | num_known_r

    timesteps = schedule.sampling_timesteps(steps)
    for t_int, s_int in itertools.pairwise(timesteps):
        t = torch.full((B * K,), t_int, dtype=torch.long, device=device)
        s = torch.full((B * K,), s_int, dtype=torch.long, device=device)
        out = net(cat_t, cat_use, num_t, num_use, clean_known, t)
        logp_x0 = masked_log_softmax(out["x0_logits"], temperature if s_int == 0 else None)
        if s_int == 0:
            new_cat = gumbel_argmax(logp_x0, generator)
        else:
            new_cat = gumbel_argmax(cat_posterior(cat_t, logp_x0.exp(), t, s, schedule, valid, card), generator)
        cat_t = torch.where(hidden_cat, new_cat, cat_t)

        if net.eps_head is not None:
            x0_hat = num_predict_x0(num_t, out["eps"].float(), t, schedule).clamp(-6.0, 6.0)
            if s_int == 0:
                new_num = x0_hat
            else:
                mean, var = num_posterior(x0_hat, num_t, t, s, schedule)
                step_noise = torch.randn(num_t.shape, generator=generator, device=device)
                new_num = mean + var.sqrt() * step_noise
            num_t = torch.where(hidden_num, new_num, num_t)

    return {"cat": cat_t.view(B, K, Cc), "num": num_t.view(B, K, -1)}


def aggregate_samples(
    cat_samples: torch.Tensor,
    num_samples: torch.Tensor,
    valid: torch.Tensor,
    cat_hidden: torch.Tensor,
    *,
    alpha: float = 0.1,
    temperature: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Summarize ``(B, K, Cc)`` samples: smoothed class frequencies, per-column mode and a medoid sample.

    The medoid is the single drawn sample that agrees most with the per-column modes on hidden columns, so
    it is always an internally consistent building (preferred for simulation inputs).
    """
    B, K, _ = cat_samples.shape
    k_max = valid.shape[-1]
    counts = torch.nn.functional.one_hot(cat_samples.clamp(min=0), k_max).sum(dim=1).float()
    n_valid = valid.sum(dim=-1).clamp(min=1).float()
    probs = (counts + alpha * valid) / (K + alpha * n_valid).view(1, -1, 1)
    probs = probs * valid
    if temperature is not None:
        probs = probs.clamp_min(1e-12) ** (1.0 / temperature.view(1, -1, 1)) * valid
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    mode = counts.argmax(dim=-1)
    agree = ((cat_samples == mode.unsqueeze(1)) & cat_hidden.unsqueeze(1)).sum(dim=-1)
    medoid_idx = agree.argmax(dim=1)
    medoid = cat_samples[torch.arange(B, device=cat_samples.device), medoid_idx]
    out = {"probs": probs, "mode": mode, "medoid": medoid, "medoid_index": medoid_idx}
    if num_samples.numel():
        out["num_mean"] = num_samples.mean(dim=1)
        out["num_std"] = num_samples.std(dim=1) if K > 1 else torch.zeros_like(out["num_mean"])
        out["num_medoid"] = num_samples[torch.arange(B, device=num_samples.device), medoid_idx]
    return out
