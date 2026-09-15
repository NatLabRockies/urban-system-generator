import itertools

import torch

from usg.models.diffusion import (
    DiffusionSchedule,
    cat_posterior,
    cat_q_xt_given_x0,
    num_posterior,
)


def _transition(a: float, k: int) -> torch.Tensor:
    """Q[i, j] = q(next = j | prev = i) for the uniform kernel."""
    return a * torch.eye(k, dtype=torch.float64) + (1 - a) / k


def test_sampling_timesteps_strided():
    sched = DiffusionSchedule(1000)
    ts = sched.sampling_timesteps(100)
    assert ts[0] == 1000 and ts[-1] == 0 and len(ts) == 101
    assert all(a > b for a, b in itertools.pairwise(ts))
    assert DiffusionSchedule(10).sampling_timesteps(100) == list(range(10, -1, -1))


def test_categorical_posterior_matches_bayes_for_any_stride():
    sched = DiffusionSchedule(50)
    k_max = 5
    card = torch.tensor([4.0, 5.0])
    valid = torch.arange(k_max).unsqueeze(0) < card.unsqueeze(1)
    ab = sched.alpha_bar
    for t_int, s_int in [(50, 30), (10, 9), (3, 1)]:
        for col, k in enumerate([4, 5]):
            for x0 in range(k):
                for xt in range(k):
                    a_ts = float(ab[t_int] / ab[s_int])
                    q_t_s = _transition(a_ts, k)[:, xt]  # q(x_t = xt | x_s = j) for all j
                    q_s_0 = _transition(float(ab[s_int]), k)[x0]  # q(x_s = j | x_0)
                    expected = q_t_s * q_s_0
                    expected = expected / expected.sum()
                    codes_t = torch.zeros(1, 2, dtype=torch.long)
                    codes_t[0, col] = xt
                    x0_probs = torch.zeros(1, 2, k_max)
                    x0_probs[0, col, x0] = 1.0
                    x0_probs[0, 1 - col, 0] = 1.0
                    logp = cat_posterior(
                        codes_t, x0_probs, torch.tensor([t_int]), torch.tensor([s_int]), sched, valid, card
                    )
                    got = logp.exp()[0, col, :k].double()
                    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
                    assert logp.exp()[0, col, k:].sum() == 0


def test_q_xt_given_x0_is_composition_of_steps():
    sched = DiffusionSchedule(20)
    k = 4
    card = torch.tensor([float(k)])
    valid = torch.ones(1, k, dtype=torch.bool)
    ab = sched.alpha_bar
    mat = torch.eye(k, dtype=torch.float64)
    for step in range(1, 21):
        mat = mat @ _transition(float(ab[step] / ab[step - 1]), k)
    for x0 in range(k):
        got = cat_q_xt_given_x0(torch.tensor([[x0]]), torch.tensor([20]), sched, valid, card).exp()[0, 0].double()
        torch.testing.assert_close(got, mat[x0], atol=1e-5, rtol=1e-4)


def test_gaussian_posterior_reduces_to_ddpm_one_step():
    sched = DiffusionSchedule(100)
    ab = sched.alpha_bar
    t_int = 40
    beta = float(1 - ab[t_int] / ab[t_int - 1])
    x0 = torch.tensor([[0.7]])
    xt = torch.tensor([[-0.2]])
    mean, var = num_posterior(x0, xt, torch.tensor([t_int]), torch.tensor([t_int - 1]), sched)
    c0 = float(ab[t_int - 1].sqrt() * beta / (1 - ab[t_int]))
    ct = float((1 - beta) ** 0.5 * (1 - ab[t_int - 1]) / (1 - ab[t_int]))
    exp_var = float((1 - ab[t_int - 1]) / (1 - ab[t_int]) * beta)
    torch.testing.assert_close(mean, torch.tensor([[c0 * 0.7 + ct * -0.2]]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(var, torch.tensor([[exp_var]]), atol=1e-7, rtol=1e-4)
    mean0, var0 = num_posterior(x0, xt, torch.tensor([t_int]), torch.tensor([0]), sched)
    torch.testing.assert_close(mean0, x0, atol=1e-6, rtol=1e-5)
    assert float(var0) == 0.0
