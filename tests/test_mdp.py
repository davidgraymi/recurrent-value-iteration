import numpy as np

from src.mdp import (
    GAMMA,
    bellman_optimality,
    exact_trajectory,
    generate_mdps,
    make_dataset,
    q_values,
    value_iteration,
)


def test_generate_shapes_ranges_and_determinism():
    ns, r = generate_mdps(50, seed=0)
    assert ns.shape == (50, 8, 3) and ns.dtype == np.int64
    assert r.shape == (50, 8, 3)
    assert ns.min() >= 0 and ns.max() <= 7
    assert r.min() >= -1 and r.max() <= 1
    ns2, r2 = generate_mdps(50, seed=0)
    assert np.array_equal(ns, ns2) and np.array_equal(r, r2)


def test_splits_are_independent():
    a = make_dataset("train", 20)
    b = make_dataset("test", 20)
    assert not np.array_equal(a["reward"], b["reward"])


def test_bellman_operator_matches_loop():
    ns, r = generate_mdps(3, seed=1)
    V = np.random.default_rng(0).normal(size=(3, 8))
    out = bellman_optimality(V, ns, r)
    for n in range(3):
        for s in range(8):
            expected = max(r[n, s, a] + GAMMA * V[n, ns[n, s, a]] for a in range(3))
            assert np.isclose(out[n, s], expected)


def test_value_iteration_converges_to_fixed_point():
    ns, r = generate_mdps(100, seed=2)
    traj, v_star, n_iters = value_iteration(ns, r)
    assert np.all(traj[:, 0] == 0)
    assert np.all(n_iters < 500)
    # V* is a fixed point of the Bellman optimality operator
    assert np.abs(bellman_optimality(v_star, ns, r) - v_star).max() < 1e-7
    # Bounded by r_max / (1 - gamma)
    assert np.abs(v_star).max() <= 1 / (1 - GAMMA) + 1e-9


def test_bellman_is_contraction_and_error_shrinks_geometrically():
    ns, r = generate_mdps(100, seed=3)
    traj, v_star, _ = value_iteration(ns, r)
    err = np.abs(traj - v_star[:, None]).max(axis=-1)  # sup-norm error per iteration
    for t in range(1, 50):
        assert np.all(err[:, t] <= GAMMA * err[:, t - 1] + 1e-6)  # V* itself is only accurate to ~1e-7
    # contraction on arbitrary pairs
    rng = np.random.default_rng(0)
    U, W = rng.normal(size=(2, 100, 8))
    lhs = np.abs(bellman_optimality(U, ns, r) - bellman_optimality(W, ns, r)).max(-1)
    assert np.all(lhs <= GAMMA * np.abs(U - W).max(-1) * (1 + 1e-9) + 1e-12)


def test_exact_trajectory_matches_value_iteration_prefix():
    ns, r = generate_mdps(10, seed=4)
    traj, _, _ = value_iteration(ns, r)
    t2 = exact_trajectory(ns, r, 20)
    assert np.allclose(traj[:, :21], t2)


def test_q_values_shape():
    ns, r = generate_mdps(4, seed=5)
    assert q_values(np.zeros((4, 8)), ns, r).shape == (4, 8, 3)
