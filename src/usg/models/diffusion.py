"""Noise schedules and forward/posterior math for mixed-type diffusion (TabDDPM).

* Numeric columns: Gaussian diffusion, ``x_t = sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps``.
* Categorical columns: multinomial diffusion with a uniform kernel (Hoogeboom et al. 2021, Kotelnikov et
  al. 2022), ``q(x_t | x_0) = Cat(ab_t * onehot(x_0) + (1 - ab_t) / K)``.

Timesteps are integers ``0..T`` with ``ab_0 = 1``. Both posteriors are written for an arbitrary earlier step
``s < t`` (``a_{t|s} = ab_t / ab_s``), which makes strided sampling exact.
Categorical tensors are padded to ``(N, C, K_max)``; ``valid`` marks real classes, ``card`` holds ``K`` per column.
"""

from __future__ import annotations

import math

import torch

from usg.models.layers import PAD_LOGIT

_LOG_EPS = 1e-30


class DiffusionSchedule:
    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        cosine_s: float = 0.008,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.timesteps = int(timesteps)
        self.schedule = schedule
        T = self.timesteps
        if schedule == "cosine":
            steps = torch.arange(T + 1, dtype=torch.float64)
            f = torch.cos(((steps / T) + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
            alpha_bar = f / f[0]
            betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        else:
            raise ValueError(f"unknown noise schedule {schedule!r}")
        alpha_bar = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(1 - betas, dim=0)])
        self.alpha_bar = alpha_bar  # float64, CPU, shape (T + 1,)
        self._cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def to_config(self) -> dict:
        return {"timesteps": self.timesteps, "schedule": self.schedule}

    def _tables(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if device not in self._cache:
            ab = self.alpha_bar
            self._cache[device] = (ab.float().to(device), (1 - ab).float().to(device))
        return self._cache[device]

    def ab(self, t: torch.Tensor) -> torch.Tensor:
        return self._tables(t.device)[0][t]

    def one_minus_ab(self, t: torch.Tensor) -> torch.Tensor:
        return self._tables(t.device)[1][t]

    def a_ts(self, t: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``a_{t|s}`` and ``1 - a_{t|s}`` computed in float64 for precision near ``s = t``."""
        ab = self.alpha_bar
        ratio = ab[t.cpu()] / ab[s.cpu()]
        return ratio.float().to(t.device), (1 - ratio).float().to(t.device)

    def sampling_timesteps(self, steps: int) -> list[int]:
        """Decreasing timesteps ``[T, ..., 0]`` with at most ``steps`` transitions."""
        steps = max(1, min(int(steps), self.timesteps))
        grid = torch.linspace(self.timesteps, 0, steps + 1).round().long().tolist()
        out: list[int] = []
        for t in grid:
            if not out or t < out[-1]:
                out.append(int(t))
        if out[-1] != 0:
            out.append(0)
        return out


# ---- categorical (multinomial) ------------------------------------------------------------------------
def _onehot(codes: torch.Tensor, k_max: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(codes.clamp(min=0), k_max).float()


def _to_log(probs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return torch.where(valid, torch.log(probs.clamp_min(_LOG_EPS)), torch.full_like(probs, PAD_LOGIT))


def cat_q_xt_given_x0(
    x0: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule, valid: torch.Tensor, card: torch.Tensor
) -> torch.Tensor:
    """Log-probabilities of ``q(x_t | x_0)``, shape ``(N, C, K_max)``."""
    k_max = valid.shape[-1]
    ab = schedule.ab(t).view(-1, 1, 1)
    probs = ab * _onehot(x0, k_max) + schedule.one_minus_ab(t).view(-1, 1, 1) / card.view(1, -1, 1)
    return _to_log(probs * valid, valid)


def cat_posterior(
    xt: torch.Tensor,
    x0_probs: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    schedule: DiffusionSchedule,
    valid: torch.Tensor,
    card: torch.Tensor,
) -> torch.Tensor:
    """Log-probabilities of ``q(x_s | x_t, x_0)`` with ``x_0`` given as (possibly soft) probabilities.

    ``q(x_s | x_t, x_0) ∝ q(x_t | x_s) · q(x_s | x_0)``; plugging predicted ``p(x_0)`` in place of the
    one-hot ``x_0`` yields the model posterior ``p_θ(x_s | x_t)``.
    """
    k_max = valid.shape[-1]
    inv_k = 1.0 / card.view(1, -1, 1)
    a_ts, one_minus_a_ts = schedule.a_ts(t, s)
    left = a_ts.view(-1, 1, 1) * _onehot(xt, k_max) + one_minus_a_ts.view(-1, 1, 1) * inv_k
    right = schedule.ab(s).view(-1, 1, 1) * x0_probs + schedule.one_minus_ab(s).view(-1, 1, 1) * inv_k
    unnorm = (left * right) * valid
    unnorm = unnorm / unnorm.sum(dim=-1, keepdim=True).clamp_min(_LOG_EPS)
    return _to_log(unnorm, valid)


def categorical_kl(log_q: torch.Tensor, log_p: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``KL(q || p)`` per ``(N, C)`` over valid classes."""
    q = log_q.exp() * valid
    return (q * (log_q - log_p)).masked_fill(~valid, 0.0).sum(dim=-1)


def uniform_log_probs(valid: torch.Tensor, card: torch.Tensor) -> torch.Tensor:
    probs = valid.float() / card.view(-1, 1)
    return _to_log(probs, valid)


# ---- numeric (Gaussian) --------------------------------------------------------------------------------
def num_q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, schedule: DiffusionSchedule) -> torch.Tensor:
    ab = schedule.ab(t).view(-1, 1)
    return ab.sqrt() * x0 + schedule.one_minus_ab(t).view(-1, 1).sqrt() * noise


def num_predict_x0(xt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule) -> torch.Tensor:
    ab = schedule.ab(t).view(-1, 1)
    return (xt - schedule.one_minus_ab(t).view(-1, 1).sqrt() * eps) / ab.sqrt()


def num_posterior(
    x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor, s: torch.Tensor, schedule: DiffusionSchedule
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and variance of ``q(x_s | x_t, x_0)`` for Gaussian diffusion."""
    ab_s = schedule.ab(s).view(-1, 1)
    one_minus_ab_t = schedule.one_minus_ab(t).view(-1, 1).clamp_min(1e-12)
    one_minus_ab_s = schedule.one_minus_ab(s).view(-1, 1)
    a_ts, one_minus_a_ts = schedule.a_ts(t, s)
    a_ts = a_ts.view(-1, 1)
    one_minus_a_ts = one_minus_a_ts.view(-1, 1)
    mean = (ab_s.sqrt() * one_minus_a_ts / one_minus_ab_t) * x0 + (a_ts.sqrt() * one_minus_ab_s / one_minus_ab_t) * xt
    var = one_minus_ab_s * one_minus_a_ts / one_minus_ab_t
    return mean, var
