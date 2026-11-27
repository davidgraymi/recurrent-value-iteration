"""Evaluate trained models on 1,000 unseen test MDPs.

Usage:
    python -m src.evaluate

Writes:
    results/metrics.npz      per-MDP, per-iteration metrics (iterations 1..96)
    results/summary.json     aggregate statistics, success criteria, outcome label
    results/report.md        human-readable report
"""

import argparse
import json
import os

import numpy as np
import torch

from src.mdp import GAMMA, exact_trajectory, make_dataset, q_values
from src.models import OneShotMLP, RecurrentValueUpdater
from src.train import CKPT_DIR, RESULTS_DIR, SEED, TRAIN_ITERS, get_device, set_seed

N_TEST = 1_000
MAX_ITERS = 96
CHECKPOINTS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96]
INTERVALS = [(16, 32), (32, 64), (64, 96)]


# ---------------------------------------------------------------- metrics


def compute_metrics(values, v_star, next_state, reward, gamma=GAMMA, tol=1e-9):
    """Metrics for value estimates.

    values : [N, T, S] estimates V_hat_t (any T; use T=1 for one-shot)
    v_star : [N, S] exact optimal values
    Returns dict of [N, T] arrays: value_error, max_value_error, regret, pct_optimal.
    """
    values = np.asarray(values, dtype=np.float64)
    N, T, S = values.shape
    abs_err = np.abs(values - v_star[:, None, :])
    value_error = abs_err.mean(axis=-1)
    max_value_error = abs_err.max(axis=-1)

    # Greedy action under the estimate: argmax_a r(s,a) + gamma * V_hat(next(s,a))
    succ = values[
        np.arange(N)[:, None, None, None],
        np.arange(T)[None, :, None, None],
        next_state[:, None, :, :],
    ]  # [N, T, S, A]
    q_hat = reward[:, None] + gamma * succ
    a_hat = q_hat.argmax(axis=-1)  # [N, T, S]

    q_star = q_values(v_star, next_state, reward, gamma)  # [N, S, A]
    q_star_t = np.broadcast_to(q_star[:, None], q_hat.shape)
    q_of_greedy = np.take_along_axis(q_star_t, a_hat[..., None], axis=-1)[..., 0]
    gap = q_star.max(axis=-1)[:, None, :] - q_of_greedy  # [N, T, S], >= 0
    regret = gap.mean(axis=-1)
    pct_optimal = 100.0 * (gap <= tol).mean(axis=-1)
    return {
        "value_error": value_error,
        "max_value_error": max_value_error,
        "regret": regret,
        "pct_optimal": pct_optimal,
    }


def pct_improving(value_error, t_from, t_to):
    """% of MDPs whose value error at t_to is strictly lower than at t_from (1-indexed)."""
    return float(100.0 * (value_error[:, t_to - 1] < value_error[:, t_from - 1]).mean())


def classify_outcome(median_err, train_iters=TRAIN_ITERS, rel_tol=0.01):
    """Label the long-horizon behaviour of a median error curve (index t-1 = iteration t).

    - instability: non-finite values, error at the end > 1.5x the best error seen
      after the training horizon, or many direction changes with non-trivial amplitude.
    - continued refinement: median error at 64 is >= rel_tol lower than at 16, and
      does not rise again by more than rel_tol from 64 to 96. If the 64 -> 96 drop is
      under 10% the label notes that the curve is levelling off at a fixed point.
    - fixed-point plateau: |relative change| from 16 to 96 below rel_tol.
    - otherwise: mixed.
    """
    tail = np.asarray(median_err[train_iters - 1 :])
    if not np.all(np.isfinite(tail)):
        return "instability (non-finite)"
    e16, e64, e96 = median_err[train_iters - 1], median_err[63], median_err[-1]
    if e96 > 1.5 * tail.min():
        return "instability (divergence)"
    d = np.diff(tail)
    signs = np.sign(d[np.abs(d) > 1e-12])
    flips = np.mean(signs[1:] != signs[:-1]) if len(signs) > 1 else 0.0
    amplitude = (tail.max() - tail.min()) / max(tail.mean(), 1e-12)
    if flips > 0.3 and amplitude > 0.05:
        return "instability (oscillation)"
    if e64 < (1 - rel_tol) * e16 and e96 <= (1 + rel_tol) * e64:
        if e96 > 0.9 * e64:
            return "continued refinement, then fixed-point plateau"
        return "continued refinement"
    if abs(e96 - e16) / e16 < rel_tol:
        return "fixed-point plateau"
    return "mixed"


def summarize(m, ts):
    """Mean/median of each metric at each iteration in `ts` (1-indexed)."""
    out = {}
    for t in ts:
        out[str(t)] = {
            f"{stat}_{k}": float(fn(v[:, t - 1]))
            for k, v in m.items()
            for stat, fn in (("mean", np.mean), ("median", np.median))
        }
    return out


# ---------------------------------------------------------------- inference


@torch.no_grad()
def recurrent_values(model, next_state, reward, n_iters, device, batch_size=1000):
    model.eval().to(device)
    out = []
    for i in range(0, len(reward), batch_size):
        ns = torch.as_tensor(next_state[i : i + batch_size], dtype=torch.long, device=device)
        r = torch.as_tensor(reward[i : i + batch_size], dtype=torch.float32, device=device)
        out.append(model(ns, r, n_iters).cpu().double().numpy())
    return np.concatenate(out)


@torch.no_grad()
def oneshot_values(model, next_state, reward, device):
    model.eval().to(device)
    ns = torch.as_tensor(next_state, dtype=torch.long, device=device)
    r = torch.as_tensor(reward, dtype=torch.float32, device=device)
    return model(ns, r).cpu().double().numpy()[:, None, :]  # [N, 1, S]


def load(model, name):
    ckpt = torch.load(os.path.join(CKPT_DIR, f"{name}.pt"), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    return model


# ---------------------------------------------------------------- report


def write_report(s, path):
    rec, ex, cr = s["recurrent"], s["exact_vi"], s["criteria"]
    lines = [
        "# Results: anytime refinement of a learned recurrent value updater",
        "",
        f"Test set: {s['n_test']} unseen MDPs. Training horizon: {TRAIN_ITERS} iterations.",
        "",
        f"**Outcome: {s['outcome']}.** Positive result for anytime refinement: "
        f"**{'yes' if cr['positive_result'] else 'no'}**.",
        "",
        "## Selected iterations",
        "",
        "| iter | median err | mean err | median max-err | median regret | mean regret "
        "| % optimal actions | exact VI median err |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in CHECKPOINTS:
        r, e = rec["per_iteration"][str(t)], ex["per_iteration"][str(t)]
        lines.append(
            f"| {t} | {r['median_value_error']:.4g} | {r['mean_value_error']:.4g} | "
            f"{r['median_max_value_error']:.4g} | {r['median_regret']:.4g} | "
            f"{r['mean_regret']:.4g} | {r['mean_pct_optimal']:.2f} | "
            f"{e['median_value_error']:.3g} |"
        )
    o = s["oneshot"]
    lines += [
        f"| one-shot | {o['median_value_error']:.4g} | {o['mean_value_error']:.4g} | "
        f"{o['median_max_value_error']:.4g} | {o['median_regret']:.4g} | "
        f"{o['mean_regret']:.4g} | {o['mean_pct_optimal']:.2f} | – |",
        "",
        "## % of test MDPs whose value error improves",
        "",
    ]
    for k, v in s["pct_improving"].items():
        lines.append(f"- iterations {k}: {v:.1f}%")
    lines += [
        "",
        "## Success criteria (16 → 64)",
        "",
        f"1. Median value error decreases: **{cr['median_value_error_decreases']}** "
        f"({cr['median_value_error_16']:.4g} → {cr['median_value_error_64']:.4g}, "
        f"{cr['value_error_change_pct']:+.1f}%)",
        f"2. Median action regret decreases: **{cr['median_regret_decreases']}** "
        f"({cr['median_regret_16']:.4g} → {cr['median_regret_64']:.4g}); "
        f"mean regret {cr['mean_regret_16']:.4g} → {cr['mean_regret_64']:.4g}",
        "3. Evaluated on unseen MDPs: **True** (independent seed for test split)",
        f"4. No collapse / divergence: **{cr['no_collapse_or_divergence']}** "
        f"(state-value std at 96: {cr['value_std_96']:.3f} vs V* std {cr['v_star_std']:.3f}; "
        f"median error at 96 = {cr['median_value_error_96']:.4g})",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_csv(m_rec, m_ex, path):
    """Mean/median of every metric at every iteration 1..96, for both methods."""
    cols, data = ["iteration"], [np.arange(1, MAX_ITERS + 1)]
    for prefix, m in (("recurrent", m_rec), ("exact_vi", m_ex)):
        for k, v in m.items():
            for stat, fn in (("mean", np.mean), ("median", np.median)):
                cols.append(f"{prefix}_{stat}_{k}")
                data.append(fn(v, axis=0))
    np.savetxt(path, np.column_stack(data), delimiter=",", header=",".join(cols),
               comments="", fmt="%.6g")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    set_seed(SEED)
    device = get_device(args.device)

    test = make_dataset("test", N_TEST)
    ns, r, v_star = test["next_state"], test["reward"], test["v_star"]
    print(f"test MDPs: {N_TEST}; exact VI converged in {test['n_iters'].min()}–"
          f"{test['n_iters'].max()} iterations")

    rec_vals = recurrent_values(load(RecurrentValueUpdater(), "recurrent"), ns, r, MAX_ITERS, device)
    one_vals = oneshot_values(load(OneShotMLP(), "oneshot"), ns, r, device)
    exact_vals = exact_trajectory(ns, r, MAX_ITERS)[:, 1:]  # V_1..V_96

    m_rec = compute_metrics(rec_vals, v_star, ns, r)
    m_ex = compute_metrics(exact_vals, v_star, ns, r)
    m_one = compute_metrics(one_vals, v_star, ns, r)

    all_t = list(range(1, MAX_ITERS + 1))
    med = lambda m, k: np.median(m[k], axis=0)  # noqa: E731
    rec_med_err = med(m_rec, "value_error")
    rec_med_reg = med(m_rec, "regret")
    outcome = classify_outcome(rec_med_err)

    value_std_96 = float(rec_vals[:, -1].std(axis=-1).mean())
    v_star_std = float(v_star.std(axis=-1).mean())
    e16, e64, e96 = rec_med_err[15], rec_med_err[63], rec_med_err[95]
    criteria = {
        "median_value_error_16": float(e16),
        "median_value_error_64": float(e64),
        "median_value_error_96": float(e96),
        "value_error_change_pct": float(100 * (e64 - e16) / e16),
        "median_value_error_decreases": bool(e64 < e16),
        "median_regret_16": float(rec_med_reg[15]),
        "median_regret_64": float(rec_med_reg[63]),
        "mean_regret_16": float(m_rec["regret"][:, 15].mean()),
        "mean_regret_64": float(m_rec["regret"][:, 63].mean()),
        "median_regret_decreases": bool(rec_med_reg[63] < rec_med_reg[15]),
        "value_std_96": value_std_96,
        "v_star_std": v_star_std,
        "no_collapse_or_divergence": bool(
            np.isfinite(rec_vals).all()
            and value_std_96 > 0.5 * v_star_std
            and not outcome.startswith("instability (divergence")
        ),
    }
    criteria["positive_result"] = bool(
        criteria["median_value_error_decreases"]
        and criteria["median_regret_decreases"]
        and criteria["no_collapse_or_divergence"]
    )

    summary = {
        "n_test": N_TEST,
        "train_iters": TRAIN_ITERS,
        "outcome": outcome,
        "criteria": criteria,
        "pct_improving": {f"{a}->{b}": pct_improving(m_rec["value_error"], a, b) for a, b in INTERVALS},
        "recurrent": {"per_iteration": summarize(m_rec, all_t)},
        "exact_vi": {"per_iteration": summarize(m_ex, all_t)},
        "oneshot": summarize(m_one, [1])["1"],
        "best_recurrent_iteration": int(np.argmin(rec_med_err) + 1),
        "best_recurrent_median_error": float(rec_med_err.min()),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "metrics.npz"),
        recurrent_values=rec_vals.astype(np.float32),
        v_star=v_star,
        **{f"recurrent_{k}": v for k, v in m_rec.items()},
        **{f"exact_{k}": v for k, v in m_ex.items()},
        **{f"oneshot_{k}": v[:, 0] for k, v in m_one.items()},
    )
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_report(summary, os.path.join(RESULTS_DIR, "report.md"))
    write_csv(m_rec, m_ex, os.path.join(RESULTS_DIR, "per_iteration.csv"))

    print(f"outcome: {outcome}; positive result: {criteria['positive_result']}")
    print("iter  median_err  median_regret  exact_median_err")
    for t in CHECKPOINTS:
        print(f"{t:4d}  {rec_med_err[t-1]:.4e}  {rec_med_reg[t-1]:.4e}     "
              f"{np.median(m_ex['value_error'][:, t-1]):.3e}")
    print(f"one-shot: median err {summary['oneshot']['median_value_error']:.4e}, "
          f"median regret {summary['oneshot']['median_regret']:.4e}")
    print("pct improving:", summary["pct_improving"])


if __name__ == "__main__":
    main()
