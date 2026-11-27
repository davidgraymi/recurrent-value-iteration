"""Generate the required plots from results/metrics.npz.

Usage:
    python -m src.plots
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluate import INTERVALS, MAX_ITERS, pct_improving  # noqa: E402
from src.train import RESULTS_DIR, TRAIN_ITERS  # noqa: E402

PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
ITERS = np.arange(1, MAX_ITERS + 1)
C_REC, C_EX, C_ONE = "#1f77b4", "#2ca02c", "#d62728"


def mark_horizon(ax):
    ax.axvline(TRAIN_ITERS, color="gray", ls="--", lw=1)
    ax.axvspan(TRAIN_ITERS, MAX_ITERS, color="gray", alpha=0.08)
    ax.text(TRAIN_ITERS + 1, 0.97, "training horizon (16)", transform=ax.get_xaxis_transform(),
            va="top", fontsize=8, color="gray")


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def band(ax, m, color, label):
    """Median line with interquartile band."""
    q25, q50, q75 = np.percentile(m, [25, 50, 75], axis=0)
    ax.plot(ITERS, q50, color=color, label=label)
    ax.fill_between(ITERS, q25, q75, color=color, alpha=0.15, lw=0)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    d = np.load(os.path.join(RESULTS_DIR, "metrics.npz"))
    rec_err, ex_err = d["recurrent_value_error"], d["exact_value_error"]
    rec_reg, ex_reg = d["recurrent_regret"], d["exact_regret"]
    one_err, one_reg = d["oneshot_value_error"], d["oneshot_regret"]

    # 1. Median value error vs iteration (log y)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    band(ax, rec_err, C_REC, "learned recurrent updater (median, IQR)")
    ax.axhline(np.median(one_err), color=C_ONE, ls=":", label="one-shot MLP (median)")
    ax.set_yscale("log")
    ax.set_xlabel("recurrent iteration t")
    ax.set_ylabel("value error  mean_s |V̂_t − V*|")
    ax.set_title("Median value error on 1,000 unseen test MDPs")
    mark_horizon(ax)
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "1_value_error.png")

    # 2. Action regret vs iteration (median and mean; median is often exactly 0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ITERS, np.median(rec_reg, axis=0), color=C_REC, label="learned recurrent (median)")
    ax.plot(ITERS, rec_reg.mean(axis=0), color=C_REC, ls="--", label="learned recurrent (mean)")
    ax.plot(ITERS, ex_reg.mean(axis=0), color=C_EX, ls="--", alpha=0.7, label="exact VI (mean)")
    ax.axhline(np.median(one_reg), color=C_ONE, ls=":", label="one-shot MLP (median)")
    ax.axhline(one_reg.mean(), color=C_ONE, ls="-.", alpha=0.6, label="one-shot MLP (mean)")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("recurrent iteration t")
    ax.set_ylabel("action regret  mean_s [max_a Q* − Q*(s, â)]")
    ax.set_title("Action regret of greedy policy")
    mark_horizon(ax)
    ax.legend(fontsize=8, loc="center right")
    save(fig, "2_action_regret.png")

    # 3. % of test MDPs whose value error improves over each interval
    intervals = [(1, 2), (2, 4), (4, 8), (8, 16)] + INTERVALS
    pct = [pct_improving(rec_err, a, b) for a, b in intervals]
    labels = [f"{a}→{b}" for a, b in intervals]
    colors = [C_EX if b <= TRAIN_ITERS else C_REC for _, b in intervals]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, pct, color=colors)
    for b_, v in zip(bars, pct):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    ax.axhline(50, color="gray", lw=0.8, ls=":")
    ax.axvline(3.5, color="gray", ls="--", lw=1)
    ax.text(3.6, 103, "beyond training horizon →", fontsize=8, color="gray")
    ax.set_ylim(0, 110)
    ax.set_xlabel("iteration interval")
    ax.set_ylabel("% of test MDPs with lower value error")
    ax.set_title("Learned updater: per-MDP improvement by interval")
    save(fig, "3_pct_improving.png")

    # 4. Exact value iteration error vs iteration
    fig, ax = plt.subplots(figsize=(7, 4.5))
    band(ax, ex_err, C_EX, "exact VI (median, IQR)")
    ax.plot(ITERS, np.median(ex_err[:, :1], axis=0) * 0.9 ** (ITERS - 1), color="k", ls=":",
            lw=1, label="γ^(t−1) × median error at t=1")
    ax.set_yscale("log")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("value error  mean_s |V_t − V*|")
    ax.set_title("Exact value iteration converges geometrically (γ = 0.9)")
    mark_horizon(ax)
    ax.legend(fontsize=8)
    save(fig, "4_exact_vi_error.png")

    # 5. Learned recurrent error vs exact VI error
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    band(ax, rec_err, C_REC, "learned recurrent (median, IQR)")
    band(ax, ex_err, C_EX, "exact VI (median, IQR)")
    ax.axhline(np.median(one_err), color=C_ONE, ls=":", label="one-shot MLP (median)")
    ax.set_yscale("log")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("value error")
    ax.set_title("Learned vs exact, by iteration")
    mark_horizon(ax)
    ax.legend(fontsize=8)
    ax = axes[1]
    for t, c in [(4, "#9ecae1"), (16, "#3182bd"), (64, "#08519c"), (96, "#08306b")]:
        ax.scatter(ex_err[:, t - 1], rec_err[:, t - 1], s=4, alpha=0.4, color=c, label=f"t={t}")
    lo = max(min(ex_err.min(), rec_err.min()), 1e-12)
    hi = max(ex_err.max(), rec_err.max())
    ax.plot([lo, hi], [lo, hi], "k:", lw=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("exact VI error at t (per MDP)")
    ax.set_ylabel("learned recurrent error at t (per MDP)")
    ax.set_title("Per-MDP: learned vs exact")
    ax.legend(fontsize=8, markerscale=3)
    save(fig, "5_learned_vs_exact.png")


if __name__ == "__main__":
    main()
