import numpy as np

from src.evaluate import classify_outcome, compute_metrics, pct_improving
from src.mdp import generate_mdps, value_iteration


def _tiny_mdp():
    """2 states, 2 actions, hand-checkable.

    state 0: a0 -> 0 with r=1 (optimal, V*=10), a1 -> 1 with r=0
    state 1: a0 -> 1 with r=0, a1 -> 0 with r=0.5
    V*(0) = 1 / (1-0.9) = 10; V*(1) = 0.5 + 0.9*10 = 9.5
    """
    ns = np.array([[[0, 1], [1, 0]]])
    r = np.array([[[1.0, 0.0], [0.0, 0.5]]])
    v_star = np.array([[10.0, 9.5]])
    return ns, r, v_star


def test_perfect_estimate_has_zero_error_and_regret():
    ns, r, v_star = _tiny_mdp()
    m = compute_metrics(v_star[:, None], v_star, ns, r)
    assert m["value_error"][0, 0] == 0 and m["max_value_error"][0, 0] == 0
    assert m["regret"][0, 0] == 0 and m["pct_optimal"][0, 0] == 100


def test_hand_computed_errors_and_regret():
    ns, r, v_star = _tiny_mdp()
    # Estimate that makes state 0 prefer a1 (go to state 1) and state 1 prefer a0 (stay).
    V_hat = np.array([[0.0, 20.0]])
    m = compute_metrics(V_hat[:, None], v_star, ns, r)
    assert np.isclose(m["value_error"][0, 0], (10 + 10.5) / 2)
    assert np.isclose(m["max_value_error"][0, 0], 10.5)
    # Q*(0,·) = [1 + 9, 0 + 8.55] -> gap 1.45 ; Q*(1,·) = [8.55, 0.5 + 9] -> gap 0.95
    assert np.isclose(m["regret"][0, 0], (1.45 + 0.95) / 2)
    assert m["pct_optimal"][0, 0] == 0


def test_exact_vi_metrics_go_to_zero():
    ns, r = generate_mdps(20, 0)
    traj, v_star, _ = value_iteration(ns, r)
    m = compute_metrics(traj[:, 1:], v_star, ns, r)
    assert m["value_error"].shape == (20, traj.shape[1] - 1)
    assert m["value_error"][:, -1].max() < 1e-7
    assert m["regret"][:, -1].max() == 0
    assert np.all(m["regret"] >= 0)


def test_pct_improving():
    err = np.array([[3.0, 2.0], [1.0, 1.0], [1.0, 2.0], [5.0, 0.0]])
    assert pct_improving(err, 1, 2) == 50.0


def test_classify_outcome():
    t = np.arange(1, 97)
    assert classify_outcome(0.9 ** t) == "continued refinement"
    levelling = np.maximum(0.9 ** t, 0.1 * 0.9 ** 40)
    assert classify_outcome(levelling) == "continued refinement, then fixed-point plateau"
    assert classify_outcome(np.where(t < 10, 1.0 / t, 0.1)) == "fixed-point plateau"
    assert classify_outcome(np.where(t < 16, 1.0, 1.0 + 0.1 * (t - 16))).startswith("instability")
    osc = np.where(t < 16, 1.0, 1.0 + 0.3 * (t % 2))
    assert classify_outcome(osc) == "instability (oscillation)"
