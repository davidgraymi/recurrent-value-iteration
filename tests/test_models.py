import numpy as np
import torch

from src.mdp import generate_mdps
from src.models import OneShotMLP, RecurrentValueUpdater


def _batch(n=5, seed=0):
    ns, r = generate_mdps(n, seed)
    return torch.as_tensor(ns), torch.as_tensor(r, dtype=torch.float32)


def test_recurrent_shapes():
    ns, r = _batch()
    m = RecurrentValueUpdater()
    out = m(ns, r, 7)
    assert out.shape == (5, 7, 8)
    assert m.step(torch.zeros(5, 8), ns, r).shape == (5, 8)


def test_recurrent_parameters_shared_across_iterations():
    m = RecurrentValueUpdater()
    # Parameter count is fixed (one MLP 2-64-64-1) and independent of n_iters.
    n_params = sum(p.numel() for p in m.parameters())
    assert n_params == (2 * 64 + 64) + (64 * 64 + 64) + (64 + 1)
    ns, r = _batch()
    # Running 10 iterations == applying step() 10 times with the same module.
    out = m(ns, r, 10)
    V = torch.zeros(5, 8)
    for t in range(10):
        V = m.step(V, ns, r)
        assert torch.allclose(out[:, t], V)
    # Changing a single shared weight changes the output of every iteration.
    with torch.no_grad():
        m.f[0].bias += 0.5
    out2 = m(ns, r, 10)
    assert torch.all((out2 - out).abs().amax(dim=-1) > 0)


def test_recurrent_gradients_reach_shared_mlp_from_late_iterations():
    m = RecurrentValueUpdater()
    ns, r = _batch()
    out = m(ns, r, 16)
    out[:, -1].sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_recurrent_with_exact_bellman_mlp_reproduces_value_iteration():
    """If f(r, v) = r + 0.9 v, the model is exact value iteration."""
    from src.mdp import exact_trajectory

    class ExactF(torch.nn.Module):
        def forward(self, x):
            return (x[..., 0] + 0.9 * x[..., 1]).unsqueeze(-1)

    class Exact(RecurrentValueUpdater):
        def __init__(self):
            super().__init__()
            self.f = ExactF()

    ns_np, r_np = generate_mdps(4, 7)
    ns, r = torch.as_tensor(ns_np), torch.as_tensor(r_np, dtype=torch.float64)
    out = Exact()(ns, r, 12).numpy()
    assert np.allclose(out, exact_trajectory(ns_np, r_np, 12)[:, 1:])


def test_oneshot_shape():
    ns, r = _batch()
    assert OneShotMLP()(ns, r).shape == (5, 8)
