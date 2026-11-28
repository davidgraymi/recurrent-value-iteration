import numpy as np
import torch

from src.coverage import coverage_of, exact_bellman_torch, make_loss, scaled_train_data
from src.mdp import bellman_optimality, exact_trajectory, generate_mdps
from src.models import RecurrentValueUpdater, ResidualValueUpdater
from src.train import to_tensors


def test_exact_bellman_torch_matches_numpy():
    ns, r = generate_mdps(6, 0)
    V = np.random.default_rng(0).uniform(-5, 5, size=(6, 8))
    out = exact_bellman_torch(torch.as_tensor(V), torch.as_tensor(ns), torch.as_tensor(r)).numpy()
    assert np.allclose(out, bellman_optimality(V, ns, r))


def test_zero_init_residual_is_exact_value_iteration():
    ns, r = generate_mdps(4, 1)
    m = ResidualValueUpdater().eval()
    out = m(torch.as_tensor(ns), torch.as_tensor(r, dtype=torch.float32), 10).detach().numpy()
    assert np.allclose(out, exact_trajectory(ns, r, 10)[:, 1:], atol=1e-5)
    m_rand = ResidualValueUpdater(zero_init=False).eval()
    out_rand = m_rand(torch.as_tensor(ns), torch.as_tensor(r, dtype=torch.float32), 10).detach().numpy()
    assert not np.allclose(out_rand, exact_trajectory(ns, r, 10)[:, 1:], atol=1e-3)


def test_residual_records_eps_penalty_only_when_training():
    ns, r = generate_mdps(3, 2)
    m = ResidualValueUpdater(zero_init=False)
    m.train()
    m(torch.as_tensor(ns), torch.as_tensor(r, dtype=torch.float32), 4)
    assert len(m.eps_penalty) == 4 and all(p > 0 for p in m.eps_penalty)
    m.eps_penalty.clear()
    m.eval()
    m(torch.as_tensor(ns), torch.as_tensor(r, dtype=torch.float32), 4)
    assert m.eps_penalty == []


def test_scaled_data_raises_coverage():
    base = scaled_train_data(1.0, n=200)
    scaled = scaled_train_data(1.3, n=200)
    assert np.array_equal(base["reward"][:100], scaled["reward"][:100])
    assert np.allclose(scaled["reward"][100:], 1.3 * base["reward"][100:])
    assert coverage_of(scaled) > coverage_of(base)
    assert coverage_of(base, op_vmax=12.0) == 12.0


def test_operator_loss_adds_a_term_and_residual_penalty_is_consumed():
    d = to_tensors(scaled_train_data(1.0, n=32), torch.device("cpu"))
    torch.manual_seed(0)
    m = RecurrentValueUpdater()
    plain = make_loss()(m, d).item()
    with_ops = make_loss(op_vmax=10.0)(m, d).item()
    assert with_ops > plain
    res = ResidualValueUpdater(zero_init=False).train()
    make_loss(residual=True)(res, d)
    assert res.eps_penalty == []
