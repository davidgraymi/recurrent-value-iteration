import numpy as np
import torch

from src.diagnose import bellman_residual, contraction_estimate, learned_T
from src.mdp import GAMMA, bellman_optimality, exact_trajectory, generate_mdps, value_iteration
from src.models import RecurrentValueUpdater


class ExactF(torch.nn.Module):
    def forward(self, x):
        return (x[..., 0] + GAMMA * x[..., 1]).unsqueeze(-1)


def exact_model():
    m = RecurrentValueUpdater()
    m.f = ExactF()
    return m


def test_bellman_residual_matches_operator_and_vanishes_at_v_star():
    ns, r = generate_mdps(10, 0)
    traj = exact_trajectory(ns, r, 5)[:, 1:]  # V_1..V_5
    res = bellman_residual(traj, ns, r)
    assert res.shape == (10, 5)
    expected = np.abs(bellman_optimality(traj[:, 2], ns, r) - traj[:, 2]).max(-1)
    assert np.allclose(res[:, 2], expected)
    _, v_star, _ = value_iteration(ns, r)
    assert bellman_residual(v_star[:, None], ns, r).max() < 1e-7


def test_learned_T_with_exact_mlp_is_bellman_operator():
    ns, r = generate_mdps(10, 1)
    V = np.random.default_rng(0).uniform(0, 5, size=(10, 8))
    out = learned_T(exact_model(), V, ns, r, torch.device("cpu"))
    assert np.allclose(out, bellman_optimality(V, ns, r), atol=1e-5)  # float32


def test_contraction_estimate_of_exact_operator_is_at_most_gamma():
    ns, r = generate_mdps(50, 2)
    _, v_star, _ = value_iteration(ns, r)
    L = contraction_estimate(exact_model(), v_star, ns, r, torch.device("cpu"), radius=0.5)
    assert L.shape == (50,)
    assert np.all(L <= GAMMA + 1e-4)
    assert np.all(L > 0)
