import numpy as np
import torch

from src.mdp import exact_trajectory, generate_mdps
from src.models import ConditionedValueUpdater
from src.residual import gamma_dataset, traj_loss, train_loss, to_tensors


def _inputs(n=4, seed=0, gamma=0.9):
    ns, r = generate_mdps(n, seed)
    return ns, r, torch.as_tensor(ns), torch.as_tensor(r, dtype=torch.float32), torch.full((n,), gamma)


def test_shapes_with_and_without_gamma_input():
    _, _, ns, r, g = _inputs()
    for gamma_input in (False, True):
        for residual in (False, True):
            m = ConditionedValueUpdater(gamma_input=gamma_input, residual=residual)
            assert m(ns, r, 5, g).shape == (4, 5, 8)
    assert ConditionedValueUpdater(gamma_input=True).f[0].in_features == 3
    assert ConditionedValueUpdater(gamma_input=False).f[0].in_features == 2


def test_correction_disabled_is_exact_vi_at_supplied_gamma():
    for gamma in (0.8, 0.95):
        ns_np, r_np, ns, r, g = _inputs(gamma=gamma)
        m = ConditionedValueUpdater(residual=True).eval()
        m.use_correction = False
        out = m(ns, r, 12, g).detach().numpy()
        assert np.allclose(out, exact_trajectory(ns_np, r_np, 12, gamma=gamma)[:, 1:], atol=1e-5)


def test_misspecified_base_ignores_supplied_gamma():
    ns_np, r_np, ns, r, g = _inputs(gamma=0.9)
    m = ConditionedValueUpdater(residual=True, base_gamma=0.8).eval()
    m.use_correction = False
    out = m(ns, r, 12, g).detach().numpy()
    assert np.allclose(out, exact_trajectory(ns_np, r_np, 12, gamma=0.8)[:, 1:], atol=1e-5)


def test_gamma_input_changes_output():
    _, _, ns, r, _ = _inputs()
    torch.manual_seed(0)
    m = ConditionedValueUpdater(gamma_input=True).eval()
    a = m(ns, r, 3, torch.full((4,), 0.8))
    b = m(ns, r, 3, torch.full((4,), 0.95))
    assert not torch.allclose(a, b)


def test_gamma_dataset_and_losses():
    d = gamma_dataset("val", 50, (0.8, 0.92))
    assert d["gamma"].min() >= 0.8 and d["gamma"].max() <= 0.92 and len(np.unique(d["gamma"])) == 50
    # each MDP's trajectory uses its own discount
    i = 3
    ref = exact_trajectory(d["next_state"][i : i + 1], d["reward"][i : i + 1], 5, gamma=d["gamma"][i])
    assert np.allclose(d["trajectory"][i, :6], ref[0])
    batch = to_tensors(d, torch.device("cpu"))
    m = ConditionedValueUpdater(gamma_input=True, residual=True).train()
    total = train_loss(m, batch).item()
    assert m.eps_penalty == []  # the penalty was consumed
    m.eval()
    assert total > traj_loss(m, batch).item()  # penalty added on top of the trajectory loss
