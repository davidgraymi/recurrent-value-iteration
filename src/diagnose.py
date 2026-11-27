"""Diagnose the learned updater's fixed point.

Usage:
    python -m src.diagnose

Notation: T is the exact Bellman optimality operator and T_hat is the learned
operator (one recurrent step of the trained model).

1. Long runs: value error out to 1,024 iterations.
2. True Bellman residual of the learned iterates, ||T(V_hat_t) - V_hat_t||, next to
   the learned operator's own residual ||T_hat(V_hat_t) - V_hat_t||.
3. Fixed-point bias: the signed error V_hat_inf - V* per state, checked for
   systematic structure.
4. One-step operator error ||T_hat(V) - T(V)|| at the same V for both operators,
   on exact-VI iterates and on learned iterates. This separates the error of a
   single learned update from the error that builds up by iterating it. Also
   estimates the learned operator's contraction factor L and checks the bound
   ||V_hat_inf - V*||_inf <= ||T_hat(V*) - V*||_inf / (1 - L).

Writes results/diagnostics/{summary.json, report.md, *.png}.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.evaluate import N_TEST, load  # noqa: E402
from src.mdp import GAMMA, bellman_optimality, exact_trajectory, make_dataset  # noqa: E402
from src.models import RecurrentValueUpdater  # noqa: E402
from src.train import RESULTS_DIR, SEED, TRAIN_ITERS, get_device, set_seed  # noqa: E402

LONG_ITERS = 1024
REPORT_ITERS = [16, 32, 64, 96, 128, 256, 512, 1024]
OUT_DIR = os.path.join(RESULTS_DIR, "diagnostics")


# ---------------------------------------------------------------- helpers


@torch.no_grad()
def learned_T(model, V, next_state, reward, device):
    """Apply the learned operator once to arbitrary value vectors V [N, S] (NumPy)."""
    out = model.step(
        torch.as_tensor(V, dtype=torch.float32, device=device),
        torch.as_tensor(next_state, dtype=torch.long, device=device),
        torch.as_tensor(reward, dtype=torch.float32, device=device),
    )
    return out.cpu().double().numpy()


def sup(x):
    """Sup-norm over the last axis."""
    return np.abs(x).max(axis=-1)


def mean_abs(x):
    return np.abs(x).mean(axis=-1)


def bellman_residual(values, next_state, reward, gamma=GAMMA):
    """||T(V_t) - V_t||_inf for values [N, T, S] -> [N, T]."""
    N, T, S = values.shape
    flat = values.transpose(1, 0, 2).reshape(T * N, S)
    ns = np.tile(next_state, (T, 1, 1))
    r = np.tile(reward, (T, 1, 1))
    TV = bellman_optimality(flat, ns, r, gamma).reshape(T, N, S).transpose(1, 0, 2)
    return sup(TV - values)


def contraction_estimate(model, centre, next_state, reward, device, radius, n_pairs=8, seed=0):
    """Largest observed ||T_hat U - T_hat W|| / ||U - W|| for random pairs near `centre`, per MDP."""
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_pairs):
        U = centre + rng.uniform(-radius, radius, size=centre.shape)
        W = centre + rng.uniform(-radius, radius, size=centre.shape)
        num = sup(learned_T(model, U, next_state, reward, device) - learned_T(model, W, next_state, reward, device))
        ratios.append(num / sup(U - W))
    return np.max(ratios, axis=0)


# ---------------------------------------------------------------- main


def main():
    set_seed(SEED)
    device = get_device("auto")
    os.makedirs(OUT_DIR, exist_ok=True)

    test = make_dataset("test", N_TEST)
    ns, r, v_star = test["next_state"], test["reward"], test["v_star"]
    model = load(RecurrentValueUpdater(), "recurrent").eval().to(device)

    # ---- 1. long run
    with torch.no_grad():
        rec = model(
            torch.as_tensor(ns, device=device),
            torch.as_tensor(r, dtype=torch.float32, device=device),
            LONG_ITERS,
        ).cpu().double().numpy()  # [N, 1024, S], V_hat_1..V_hat_1024
    exact = exact_trajectory(ns, r, LONG_ITERS)[:, 1:]
    rec_err = mean_abs(rec - v_star[:, None])  # [N, T]
    ex_err = mean_abs(exact - v_star[:, None])

    # ---- 2. residuals
    true_resid = bellman_residual(rec, ns, r)  # ||T V_hat_t - V_hat_t||_inf
    self_resid = sup(np.diff(rec, axis=1))  # ||T_hat V_hat_t - V_hat_t||_inf, t = 1..1023
    v_inf = rec[:, -1]
    # Standard bound: ||V - V*||_inf <= ||T V - V||_inf / (1 - gamma)
    bound_ok = float((sup(v_inf - v_star) <= true_resid[:, -1] / (1 - GAMMA) + 1e-6).mean())

    # ---- 3. fixed-point bias
    bias = v_inf - v_star  # [N, S] signed
    b = bias.ravel()
    vs = v_star.ravel()
    # Bias as a function of V*, in 1-unit bins (the relation is not linear).
    bin_edges = np.arange(np.floor(vs.min()), 11)
    bin_idx = np.digitize(vs, bin_edges) - 1
    bias_by_vstar = [
        {"v_star_bin": [float(bin_edges[i]), float(bin_edges[i] + 1)],
         "n_states": int((bin_idx == i).sum()),
         "mean_bias": float(b[bin_idx == i].mean())}
        for i in range(len(bin_edges)) if (bin_idx == i).sum() > 0
    ]
    # Coverage: successor values the MLP actually saw in training (V_0..V_15).
    train_inputs = make_dataset("train", 10_000)["trajectory"][:, :TRAIN_ITERS]
    train_max = float(train_inputs.max())
    train_p99 = float(np.percentile(train_inputs, 99))
    high = v_star.max(axis=1) > train_max  # MDPs whose V* leaves the training range
    sup_err = sup(bias)
    per_mdp_mean = bias.mean(axis=1)
    within_std = bias.std(axis=1)
    sign_consistent = float(((bias > 0).all(1) | (bias < 0).all(1)).mean())
    # how much of the per-state bias is a constant per-MDP offset?
    frac_offset = float((per_mdp_mean ** 2).mean() / (bias ** 2).mean())

    # The scalar map f_theta(r, v) against the true r + gamma v on a grid
    r_grid = np.linspace(-1, 1, 101)
    v_grid = np.linspace(-10, 10, 201)
    R, Vg = np.meshgrid(r_grid, v_grid, indexing="ij")
    with torch.no_grad():
        x = torch.as_tensor(np.stack([R, Vg], -1), dtype=torch.float32, device=device)
        f = model.f(x).squeeze(-1).cpu().double().numpy()
    f_err = f - (R + GAMMA * Vg)
    # Effective discount: slope of f in v
    eff_gamma = np.gradient(f, v_grid, axis=1)
    v_lo, v_hi = np.percentile(v_star, [1, 99])
    in_range = (Vg >= v_lo) & (Vg <= v_hi)
    seen = in_range & (Vg <= train_max)  # covered by training inputs
    unseen = in_range & (Vg > train_max)  # needed at the fixed point, never trained on

    # ---- 4. one-step operator error at the same inputs
    op_err = {}
    for t in [0, 4, 16, 32, 64]:
        V = exact[:, t - 1] if t > 0 else np.zeros_like(v_star)
        op_err[f"exact V_{t}"] = sup(learned_T(model, V, ns, r, device) - bellman_optimality(V, ns, r))
    op_err["V*"] = sup(learned_T(model, v_star, ns, r, device) - v_star)
    for t in [16, 64, 1024]:
        V = rec[:, t - 1]
        op_err[f"learned V_hat_{t}"] = sup(learned_T(model, V, ns, r, device) - bellman_optimality(V, ns, r))

    # Error decomposition: ||V_hat_t - V_t|| (gap between the two trajectories)
    traj_gap = sup(rec - exact)

    # Contraction factor of T_hat near V* and the fixed-point bound
    L = contraction_estimate(model, v_star, ns, r, device, radius=0.5)
    one_step_at_star = op_err["V*"]
    fp_bound = np.where(L < 1, one_step_at_star / np.maximum(1 - L, 1e-6), np.inf)
    fp_actual = sup(v_inf - v_star)
    # Empirical late-stage convergence rate of the learned iteration
    late_rate = self_resid[:, 60] / np.maximum(self_resid[:, 59], 1e-12)  # t=200 is already an exact float32 fixed point

    def q(x):
        return {"median": float(np.median(x)), "mean": float(np.mean(x)),
                "p90": float(np.percentile(x, 90)), "max": float(np.max(x))}

    summary = {
        "long_run": {
            str(t): {
                "learned_median_value_error": float(np.median(rec_err[:, t - 1])),
                "learned_mean_value_error": float(rec_err[:, t - 1].mean()),
                "exact_median_value_error": float(np.median(ex_err[:, t - 1])),
                "learned_median_true_bellman_residual": float(np.median(true_resid[:, t - 1])),
                "learned_median_self_residual": float(np.median(self_resid[:, t - 1])) if t < LONG_ITERS else float(np.median(sup(learned_T(model, v_inf, ns, r, device) - v_inf))),
                "median_gap_to_exact_trajectory": float(np.median(traj_gap[:, t - 1])),
            }
            for t in REPORT_ITERS
        },
        "pct_mdps_error_rises_96_to_1024": float(100 * (rec_err[:, -1] > rec_err[:, 95]).mean()),
        "residual_bound_holds_fraction": bound_ok,
        "bias": {
            "mean_signed": float(b.mean()),
            "mean_abs": float(np.abs(b).mean()),
            "pct_states_overestimated": float(100 * (b > 0).mean()),
            "mean_bias_by_vstar_bin": bias_by_vstar,
            "pct_mdps_same_sign_all_states": 100 * sign_consistent,
            "fraction_of_squared_bias_that_is_per_mdp_offset": frac_offset,
            "median_within_mdp_std": float(np.median(within_std)),
        },
        "training_coverage": {
            "max_successor_value_seen_in_training": train_max,
            "p99_successor_value_seen_in_training": train_p99,
            "pct_test_vstar_entries_above_train_max": float(100 * (vs > train_max).mean()),
            "pct_test_mdps_with_any_vstar_above_train_max": float(100 * high.mean()),
            "median_sup_fixed_point_error_mdps_inside_range": float(np.median(sup_err[~high])),
            "median_sup_fixed_point_error_mdps_leaving_range": float(np.median(sup_err[high])),
        },
        "f_theta_vs_bellman_on_grid": {
            "v_range_used(1st-99th pct of V*)": [float(v_lo), float(v_hi)],
            "mean_abs_error_seen_range": float(np.abs(f_err[seen]).mean()),
            "mean_abs_error_unseen_range": float(np.abs(f_err[unseen]).mean()),
            "mean_signed_error_unseen_range": float(f_err[unseen].mean()),
            "max_abs_error_in_range": float(np.abs(f_err[in_range]).max()),
            "effective_gamma_seen_range_median": float(np.median(eff_gamma[seen])),
            "effective_gamma_unseen_range_median": float(np.median(eff_gamma[unseen])),
        },
        "one_step_operator_error_sup": {k: q(v) for k, v in op_err.items()},
        "contraction": {
            "L_near_vstar": q(L),
            "late_iteration_rate_t60": q(late_rate),
            "pct_mdps_L_lt_1": float(100 * (L < 1).mean()),
            "fixed_point_error_sup": q(fp_actual),
            "bound_one_step_over_1_minus_L": q(fp_bound[np.isfinite(fp_bound)]),
            "bound_holds_fraction": float((fp_actual <= fp_bound + 1e-5).mean()),
        },
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---------------------------------------------------------------- plots
    T = np.arange(1, LONG_ITERS + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(T, np.median(rec_err, 0), label="learned: value error mean_s|V̂−V*|")
    ax.plot(T, np.median(ex_err, 0), label="exact VI: value error", color="#2ca02c")
    ax.plot(T, np.median(true_resid, 0), label="learned: true Bellman residual ‖T V̂−V̂‖∞", ls="--")
    ax.plot(T[:-1], np.median(self_resid, 0), label="learned: self residual ‖T̂ V̂−V̂‖∞", ls=":")
    ax.axvline(TRAIN_ITERS, color="gray", ls="--", lw=1)
    ax.text(TRAIN_ITERS * 1.05, 0.97, "training horizon", transform=ax.get_xaxis_transform(),
            va="top", fontsize=8, color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 20)
    ax.set_xlabel("iteration t")
    ax.set_ylabel("median over 1,000 test MDPs")
    ax.set_title("Long run: learned operator converges, but not to V*")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "1_long_run_residuals.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.scatter(vs, b, s=2, alpha=0.25, label="states (test MDPs)")
    ok = [d for d in bias_by_vstar if d["n_states"] >= 20]
    ax.plot([d["v_star_bin"][0] + 0.5 for d in ok], [d["mean_bias"] for d in ok], "ro-", ms=3, lw=1,
            label="mean bias per unit bin of V*")
    ax.axvline(train_max, color="k", ls="--", lw=1, label=f"max successor value in training ({train_max:.2f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(-1, 10.2)
    ax.set_ylim(-0.9, 0.3)
    ax.set_xlabel("V*(s)")
    ax.set_ylabel("bias V̂₁₀₂₄(s) − V*(s)")
    ax.set_title("Fixed-point bias per state")
    ax.legend(fontsize=8)
    ax = axes[1]
    im = ax.pcolormesh(v_grid, r_grid, f_err, cmap="RdBu_r", shading="auto",
                       vmin=-np.abs(f_err[in_range]).max(), vmax=np.abs(f_err[in_range]).max())
    ax.axvspan(v_grid[0], v_lo, color="white", alpha=0.6)
    ax.axvspan(v_hi, v_grid[-1], color="white", alpha=0.6)
    ax.axvline(train_max, color="k", ls="--", lw=1)
    ax.text(train_max - 0.2, 0.9, "training\nmax", ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, label="f_θ(r, v) − (r + 0.9 v)")
    ax.set_xlabel("successor value v")
    ax.set_ylabel("reward r")
    ax.set_title("Learned update error (unshaded = 1–99% range of V*)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "2_fixed_point_bias.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = list(op_err)
    ax.boxplot([op_err[k] for k in labels], showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=30, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("‖T̂(V) − T(V)‖∞ per MDP")
    ax.set_title("One-step error of the learned operator at the same input V")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "3_one_step_operator_error.png"), dpi=150)
    plt.close(fig)

    write_report(summary)
    print(json.dumps(summary, indent=1))


def write_report(s):
    lr, bias, fg, op, c, cov = (s["long_run"], s["bias"], s["f_theta_vs_bellman_on_grid"],
                                s["one_step_operator_error_sup"], s["contraction"], s["training_coverage"])
    lines = [
        "# Fixed-point diagnostics of the learned recurrent updater",
        "",
        "## 1–2. Long run and Bellman residuals (medians over 1,000 test MDPs)",
        "",
        "| iter | learned value error | exact VI value error | true residual ‖TV̂−V̂‖∞ | self residual ‖T̂V̂−V̂‖∞ | gap to exact trajectory ‖V̂_t−V_t‖∞ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for t, v in lr.items():
        lines.append(f"| {t} | {v['learned_median_value_error']:.4g} | {v['exact_median_value_error']:.3g} | "
                     f"{v['learned_median_true_bellman_residual']:.3g} | {v['learned_median_self_residual']:.3g} | "
                     f"{v['median_gap_to_exact_trajectory']:.3g} |")
    lines += [
        "",
        f"Error rises from 96 to 1,024 on {s['pct_mdps_error_rises_96_to_1024']:.1f}% of MDPs.",
        "",
        "## 3. Fixed-point bias (V̂₁₀₂₄ − V*)",
        "",
        f"- Mean signed bias {bias['mean_signed']:+.4f}, mean absolute bias {bias['mean_abs']:.4f}; "
        f"{bias['pct_states_overestimated']:.1f}% of states are overestimated.",
        f"- {bias['pct_mdps_same_sign_all_states']:.1f}% of MDPs have the same bias sign on all 8 states; "
        f"a per-MDP constant offset explains {100 * bias['fraction_of_squared_bias_that_is_per_mdp_offset']:.1f}% "
        f"of the squared bias.",
        "",
        "| V* bin | states | mean bias |",
        "|---|---:|---:|",
        *[f"| [{d['v_star_bin'][0]:.0f}, {d['v_star_bin'][1]:.0f}) | {d['n_states']} | {d['mean_bias']:+.4f} |"
          for d in bias["mean_bias_by_vstar_bin"]],
        "",
        "### Training coverage",
        "",
        f"- Largest successor value the MLP saw in training (V_0..V_15): {cov['max_successor_value_seen_in_training']:.2f} "
        f"(99th percentile {cov['p99_successor_value_seen_in_training']:.2f}).",
        f"- {cov['pct_test_vstar_entries_above_train_max']:.1f}% of test V* entries, and "
        f"{cov['pct_test_mdps_with_any_vstar_above_train_max']:.1f}% of test MDPs, go above that.",
        f"- Median fixed-point error ‖V̂∞ − V*‖∞: {cov['median_sup_fixed_point_error_mdps_inside_range']:.4f} for MDPs that stay "
        f"inside the training range vs {cov['median_sup_fixed_point_error_mdps_leaving_range']:.4f} for MDPs that leave it.",
        f"- f_θ − (r + 0.9v) over the V* range: mean |error| {fg['mean_abs_error_seen_range']:.4f} where training covered v, "
        f"{fg['mean_abs_error_unseen_range']:.4f} above the training max (mean signed {fg['mean_signed_error_unseen_range']:+.4f}). "
        f"Effective γ (∂f/∂v): {fg['effective_gamma_seen_range_median']:.3f} seen vs "
        f"{fg['effective_gamma_unseen_range_median']:.3f} unseen.",
        "",
        "## 4. One-step operator error ‖T̂(V) − T(V)‖∞",
        "",
        "| input V | median | p90 |",
        "|---|---:|---:|",
    ]
    for k, v in op.items():
        lines.append(f"| {k} | {v['median']:.4g} | {v['p90']:.4g} |")
    lines += [
        "",
        f"Contraction factor of T̂ near V*: median {c['L_near_vstar']['median']:.3f}, "
        f"max {c['L_near_vstar']['max']:.3f} ({c['pct_mdps_L_lt_1']:.1f}% of MDPs below 1). "
        f"Late-iteration convergence rate at t=60: median {c['late_iteration_rate_t60']['median']:.3f}.",
        "",
        f"Fixed-point error ‖V̂∞ − V*‖∞: median {c['fixed_point_error_sup']['median']:.4f}; "
        f"bound ‖T̂V* − V*‖∞ / (1 − L): median {c['bound_one_step_over_1_minus_L']['median']:.4f}; "
        f"bound holds on {100 * c['bound_holds_fraction']:.1f}% of MDPs.",
        "",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
