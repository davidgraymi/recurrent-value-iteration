"""Does broader value coverage in training remove the learned fixed-point bias?

Usage:
    python -m src.coverage                         # all variants, seeds 42-44 (~15-20 min on CPU)
    python -m src.coverage --variants baseline residual --seeds 42
    python -m src.coverage --ci                    # bootstrap CIs from saved checkpoints

Every variant keeps the same test MDPs, the same 2-64-64-1 update MLP and the
same training configuration as src/train.py (10k training MDPs, batch 128,
Adam 1e-3, up to 100 epochs, early stopping on the original validation
trajectory loss). Only what the updater is trained on, or how f is
parameterised, changes:

  baseline          exact V_1..V_16 trajectories (successor inputs reach 7.94)
  broad_ops_<a>     baseline + a one-step operator loss on random value vectors
                    V ~ U[-a, a]^8 with target T(V)   (a = 9, 10, 12)
  reward_scale_1.3  half the training MDPs have rewards x1.3, so 16-step
                    trajectories reach successor values up to ~10.3
  residual          f(r, v) = r + 0.9 v + eps(r, v); eps starts at 0 and gets an
                    L2 penalty (weight 1e-3) on every input it sees; trained on
                    baseline trajectories only
  residual_randinit as residual, but eps starts as a random MLP (default init), so
                    training must remove it; tests what happens to eps outside
                    the trained range

Writes results/coverage/{summary.json, report.md, *.png}; checkpoints go to
checkpoints/coverage/.
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.diagnose import bellman_residual, learned_T, mean_abs, sup  # noqa: E402
from src.evaluate import N_TEST, compute_metrics  # noqa: E402
from src.mdp import GAMMA, exact_trajectory, make_dataset, value_iteration  # noqa: E402
from src.models import RecurrentValueUpdater, ResidualValueUpdater  # noqa: E402
from src.train import CKPT_DIR, RESULTS_DIR, TRAIN_ITERS, recurrent_loss, to_tensors, train_model  # noqa: E402

OUT_DIR = os.path.join(RESULTS_DIR, "coverage")
CKPT_SUBDIR = os.path.join(CKPT_DIR, "coverage")
LONG_ITERS = 1024
EPS_WEIGHT = 1e-3
VARIANTS = {
    "baseline": {},
    "broad_ops_9": {"op_vmax": 9.0},
    "broad_ops_10": {"op_vmax": 10.0},
    "broad_ops_12": {"op_vmax": 12.0},
    "reward_scale_1.3": {"reward_scale": 1.3},
    "residual": {"residual": True},
    "residual_randinit": {"residual": True, "zero_init": False},
}
DEFAULT_SEEDS = [42, 43, 44]


# ---------------------------------------------------------------- data


def scaled_train_data(scale, n=10_000):
    """The standard training MDPs with the rewards of the second half multiplied by `scale`."""
    d = make_dataset("train", n)
    reward = d["reward"].copy()
    reward[n // 2 :] *= scale
    trajectory, v_star, n_iters = value_iteration(d["next_state"], reward)
    return {"next_state": d["next_state"], "reward": reward, "trajectory": trajectory,
            "v_star": v_star, "n_iters": n_iters}


def coverage_of(train_np, op_vmax=None):
    """Largest successor value the update MLP is trained on."""
    traj_max = float(train_np["trajectory"][:, :TRAIN_ITERS].max())
    return max(traj_max, op_vmax or -np.inf)


# ---------------------------------------------------------------- losses


def exact_bellman_torch(V, next_state, reward, gamma=GAMMA):
    """Exact T(V) in torch, used only as a training target."""
    B, S, A = next_state.shape
    succ = torch.gather(V, 1, next_state.reshape(B, S * A)).reshape(B, S, A)
    return (reward + gamma * succ).max(dim=-1).values


def make_loss(op_vmax=None, residual=False, seed=0):
    """Trajectory loss, plus an optional random-operator loss and eps penalty."""
    gen = torch.Generator().manual_seed(seed + 1)

    def loss_fn(model, batch):
        loss = recurrent_loss(model, batch)
        if op_vmax is not None:
            B, S = batch["reward"].shape[:2]
            V = (torch.rand(B, S, generator=gen) * 2 - 1).to(batch["reward"].device) * op_vmax
            target = exact_bellman_torch(V, batch["next_state"], batch["reward"])
            loss = loss + ((model.step(V, batch["next_state"], batch["reward"]) - target) ** 2).mean()
        if residual and model.training and model.eps_penalty:
            loss = loss + EPS_WEIGHT * torch.stack(model.eps_penalty).mean()
            model.eps_penalty.clear()
        return loss

    return loss_fn


# ---------------------------------------------------------------- diagnostics


@torch.no_grad()
def diagnose_model(model, test, base_max, device):
    """Long-run fixed-point metrics on the test set."""
    model.eval().to(device)
    ns, r, v_star = test["next_state"], test["reward"], test["v_star"]
    rec = model(
        torch.as_tensor(ns, device=device),
        torch.as_tensor(r, dtype=torch.float32, device=device),
        LONG_ITERS,
    ).cpu().double().numpy()
    err = mean_abs(rec - v_star[:, None])
    v_inf = rec[:, -1]
    fp_sup = sup(v_inf - v_star)
    high = v_star.max(axis=1) > base_max
    one_step_star = sup(learned_T(model, v_star, ns, r, device) - v_star)
    m = compute_metrics(rec[:, [15, 63, LONG_ITERS - 1]], v_star, ns, r)

    r_grid = np.linspace(-1, 1, 41)
    R_lo, V_lo = np.meshgrid(r_grid, np.linspace(0, base_max, 81), indexing="ij")
    R_hi, V_hi = np.meshgrid(r_grid, np.linspace(base_max, 10, 41), indexing="ij")

    def f_err(R, Vg):
        x = torch.as_tensor(np.stack([R, Vg], -1), dtype=torch.float32, device=device)
        return np.abs(model.q(x).cpu().double().numpy() - (R + GAMMA * Vg)).mean()

    return {
        "median_value_error": {str(t): float(np.median(err[:, t - 1])) for t in (16, 64, 256, LONG_ITERS)},
        "median_true_bellman_residual_final": float(np.median(bellman_residual(v_inf[:, None], ns, r))),
        "median_self_residual_final": float(np.median(sup(learned_T(model, v_inf, ns, r, device) - v_inf))),
        "median_fixed_point_error_sup": float(np.median(fp_sup)),
        "median_fixed_point_error_sup_low_mdps": float(np.median(fp_sup[~high])),
        "median_fixed_point_error_sup_high_mdps": float(np.median(fp_sup[high])),
        "median_one_step_error_at_vstar": float(np.median(one_step_star)),
        "median_one_step_error_at_vstar_high_mdps": float(np.median(one_step_star[high])),
        "f_abs_error_v_below_base_max": float(f_err(R_lo, V_lo)),
        "f_abs_error_v_above_base_max": float(f_err(R_hi, V_hi)),
        "mean_regret": {k: float(m["regret"][:, i].mean()) for i, k in enumerate(["16", "64", str(LONG_ITERS)])},
        "pct_optimal_final": float(m["pct_optimal"][:, -1].mean()),
        "median_error_curve": np.median(err, axis=0).tolist(),
    }


# ---------------------------------------------------------------- report


METRICS = [
    ("coverage", "train coverage (max v)", "{:.2f}"),
    ("median_fixed_point_error_sup", "fixed-pt err ‖V̂∞−V*‖∞", "{:.3g}"),
    ("median_fixed_point_error_sup_high_mdps", "… high-V* MDPs", "{:.3g}"),
    ("median_true_bellman_residual_final", "true residual ‖TV̂∞−V̂∞‖∞", "{:.3g}"),
    ("median_one_step_error_at_vstar_high_mdps", "one-step err at V* (high MDPs)", "{:.3g}"),
    ("f_abs_error_v_above_base_max", "|f − (r+0.9v)|, v > 7.94", "{:.3g}"),
    ("f_abs_error_v_below_base_max", "|f − (r+0.9v)|, v ≤ 7.94", "{:.3g}"),
    ("value_error_16", "median value err @16", "{:.3g}"),
    ("value_error_final", "median value err @1024", "{:.3g}"),
    ("mean_regret_final", "mean regret @1024", "{:.3g}"),
    ("pct_optimal_final", "% optimal actions @1024", "{:.2f}"),
    ("epochs_run", "epochs run", "{:.0f}"),
]


def flat(run):
    d = dict(run)
    d["value_error_16"] = run["median_value_error"]["16"]
    d["value_error_final"] = run["median_value_error"][str(LONG_ITERS)]
    d["mean_regret_final"] = run["mean_regret"][str(LONG_ITERS)]
    return d


def aggregate(runs):
    out = {}
    for key, _, _ in METRICS:
        vals = np.array([flat(r)[key] for r in runs])
        out[key] = {"mean": float(vals.mean()), "std": float(vals.std()), "per_seed": vals.tolist()}
    return out


def write_report(summary, path):
    names = list(summary["variants"])
    lines = [
        "# Training coverage vs the learned fixed point",
        "",
        f"Test set: {N_TEST} unseen MDPs, {LONG_ITERS} iterations. Seeds: {summary['seeds']}. "
        f"Values are the mean ± std over seeds of each run's median over test MDPs. "
        f"\"High-V* MDPs\" are those with some V*(s) above {summary['base_max']:.2f}, the largest successor "
        f"value in the baseline training trajectories ({summary['pct_high_mdps']:.1f}% of test MDPs). "
        f"Exact VI at {LONG_ITERS} iterations: median value error {summary['exact_vi_final_error']:.1e}.",
        "",
        "| metric | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for key, label, fmt in METRICS:
        cells = []
        for n in names:
            a = summary["variants"][n]["aggregate"][key]
            cells.append(fmt.format(a["mean"]) + (f" ± {fmt.format(a['std'])}" if a["std"] > 0 and key not in ("coverage", "epochs_run") else ""))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def plot(summary, exact_curve):
    names = list(summary["variants"])
    colors = dict(zip(names, plt.rcParams["axes.prop_cycle"].by_key()["color"]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    panels = [("median_fixed_point_error_sup", "all test MDPs"),
              ("median_fixed_point_error_sup_high_mdps", f"high-V* MDPs (some V* > {summary['base_max']:.2f})")]
    for ax, (key, title) in zip(axes, panels):
        for n in names:
            a = summary["variants"][n]["aggregate"]
            marker = "s" if n.startswith("residual") else "o"
            ax.scatter(a["coverage"]["per_seed"], a[key]["per_seed"],
                       color=colors[n], marker=marker, label=n, s=30, alpha=0.8)
        ax.axvline(10, color="gray", ls=":", lw=1)
        ax.text(10.05, 0.97, "max V* = 10", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color="gray")
        ax.axvline(summary["base_max"], color="gray", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("training coverage: largest successor value the update MLP is trained on")
        ax.set_ylabel("median ‖V̂₁₀₂₄ − V*‖∞")
        ax.set_title(f"Fixed-point error vs coverage: {title}")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "1_fixed_point_error_vs_coverage.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    T = np.arange(1, LONG_ITERS + 1)
    for n in names:
        curves = np.array([r["median_error_curve"] for r in summary["variants"][n]["runs"]])
        ax.plot(T, curves.mean(0), color=colors[n], label=n)
    ax.plot(T, exact_curve, color="k", ls="--", lw=1, label="exact VI")
    ax.axvline(TRAIN_ITERS, color="gray", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 20)
    ax.set_xlabel("iteration t")
    ax.set_ylabel("median value error (mean over seeds)")
    ax.set_title("Long-run value error by training variant")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "2_long_run_by_variant.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    keys = [("f_abs_error_v_above_base_max", "|f − (r+0.9v)|, v > 7.94"),
            ("median_one_step_error_at_vstar_high_mdps", "one-step err at V* (high MDPs)"),
            ("median_true_bellman_residual_final", "true Bellman residual at 1024")]
    w = 0.8 / len(keys)
    x = np.arange(len(names))
    for i, (k, label) in enumerate(keys):
        means = [summary["variants"][n]["aggregate"][k]["mean"] for n in names]
        stds = [summary["variants"][n]["aggregate"][k]["std"] for n in names]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, label=label, capsize=2)
    ax.set_xticks(x, names, rotation=20, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("error (mean ± std over seeds)")
    ax.set_title("High-value operator error and asymptotic Bellman residual")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "3_operator_error_by_variant.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- bootstrap CIs


def bootstrap_ci(x, stat=np.median, n_boot=2000, seed=0, level=95):
    """Percentile bootstrap CI of stat(x), resampling entries of x."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    lo, hi = np.percentile(boots, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(stat(x)), float(lo), float(hi)


def paired_ratio_ci(x, y, n_boot=2000, seed=0, level=95):
    """CI of median(x) / median(y), resampling MDPs jointly (x, y measured on the same MDPs)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.median(x[idx], axis=1) / np.median(y[idx], axis=1)
    lo, hi = np.percentile(boots, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(np.median(x) / np.median(y)), float(lo), float(hi)


@torch.no_grad()
def per_mdp_fixed_point_error(model, test, device):
    """Per-MDP ||V_hat_1024 - V*||_inf and mean regret at 1024."""
    model.eval().to(device)
    ns, r, v_star = test["next_state"], test["reward"], test["v_star"]
    V = torch.zeros(len(r), ns.shape[1], device=device)
    ns_t = torch.as_tensor(ns, device=device)
    r_t = torch.as_tensor(r, dtype=torch.float32, device=device)
    for _ in range(LONG_ITERS):
        V = model.step(V, ns_t, r_t)
    v_inf = V.cpu().double().numpy()
    regret = compute_metrics(v_inf[:, None], v_star, ns, r)["regret"][:, 0]
    return sup(v_inf - v_star), regret


def confidence_intervals(test, base_max, device, seeds, variants):
    """Bootstrap 95% CIs over test MDPs for each variant's seed-averaged per-MDP errors."""
    high = test["v_star"].max(axis=1) > base_max
    per = {}
    for name in variants:
        errs, regs = [], []
        for seed in seeds:
            cfg = VARIANTS[name]
            if cfg.get("residual"):
                model = ResidualValueUpdater(zero_init=cfg.get("zero_init", True))
            else:
                model = RecurrentValueUpdater()
            ckpt = torch.load(os.path.join(CKPT_SUBDIR, f"{name}_s{seed}.pt"), map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            e, rg = per_mdp_fixed_point_error(model, test, device)
            errs.append(e)
            regs.append(rg)
        per[name] = (np.mean(errs, axis=0), np.mean(regs, axis=0))
    out = {"n_mdps": int(len(high)), "n_high_mdps": int(high.sum()), "seeds": seeds, "variants": {}}
    base_err = per["baseline"][0]
    for name, (err, reg) in per.items():
        out["variants"][name] = {
            "median_fp_error_all": bootstrap_ci(err),
            "median_fp_error_high": bootstrap_ci(err[high]),
            "mean_regret_all": bootstrap_ci(reg, stat=np.mean),
            "ratio_to_baseline_all": paired_ratio_ci(err, base_err),
            "ratio_to_baseline_high": paired_ratio_ci(err[high], base_err[high]),
        }
    return out


def write_ci_report(ci, path):
    fmt = lambda t: f"{t[0]:.3g} [{t[1]:.3g}, {t[2]:.3g}]"  # noqa: E731
    lines = [
        "# Bootstrap confidence intervals (coverage experiment)",
        "",
        f"95% percentile bootstrap over test MDPs (2,000 resamples). Each MDP's error is first averaged "
        f"over seeds {ci['seeds']}, so these intervals reflect variation across MDPs, not across seeds. "
        f"All MDPs: n = {ci['n_mdps']}; high-V* MDPs: n = {ci['n_high_mdps']}. "
        f"Ratios are paired (same MDPs resampled for both variants); a ratio interval that excludes 1 "
        f"is a difference from baseline beyond MDP-sampling noise.",
        "",
        "| variant | median ‖V̂∞−V*‖∞, all | ratio to baseline, all | median ‖V̂∞−V*‖∞, high-V* | ratio to baseline, high-V* | mean regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for n, v in ci["variants"].items():
        lines.append(f"| {n} | {fmt(v['median_fp_error_all'])} | {fmt(v['ratio_to_baseline_all'])} | "
                     f"{fmt(v['median_fp_error_high'])} | {fmt(v['ratio_to_baseline_high'])} | "
                     f"{fmt(v['mean_regret_all'])} |")
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--device", default="cpu", help="CPU is deterministic and as fast for this model")
    p.add_argument("--replot", action="store_true", help="redraw plots and report from summary.json")
    p.add_argument("--ci", action="store_true", help="bootstrap CIs from saved checkpoints")
    args = p.parse_args()
    device = torch.device(args.device)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_SUBDIR, exist_ok=True)
    targs = SimpleNamespace(lr=1e-3, epochs=100, batch_size=128, patience=10)

    base_np = make_dataset("train", 10_000)
    base_max = coverage_of(base_np)
    base_train = to_tensors(base_np, device)
    val = to_tensors(make_dataset("val", 1_000), device)
    test = make_dataset("test", N_TEST)
    exact_curve = np.median(
        mean_abs(exact_trajectory(test["next_state"], test["reward"], LONG_ITERS)[:, 1:] - test["v_star"][:, None]),
        axis=0,
    )

    if args.ci:
        ci = confidence_intervals(test, base_max, device, args.seeds, args.variants)
        with open(os.path.join(OUT_DIR, "ci.json"), "w") as fh:
            json.dump(ci, fh, indent=1)
        write_ci_report(ci, os.path.join(OUT_DIR, "ci.md"))
        print(open(os.path.join(OUT_DIR, "ci.md")).read())
        return

    if args.replot:
        with open(os.path.join(OUT_DIR, "summary.json")) as fh:
            summary = json.load(fh)
        write_report(summary, os.path.join(OUT_DIR, "report.md"))
        plot(summary, exact_curve)
        return

    summary = {"seeds": args.seeds, "base_max": base_max,
               "pct_high_mdps": float(100 * (test["v_star"].max(1) > base_max).mean()),
               "exact_vi_final_error": float(exact_curve[-1]), "variants": {}}
    t_all = time.time()
    for name in args.variants:
        cfg = VARIANTS[name]
        if "reward_scale" in cfg:
            train_np = scaled_train_data(cfg["reward_scale"])
            train = to_tensors(train_np, device)
        else:
            train_np, train = base_np, base_train
        coverage = coverage_of(train_np, cfg.get("op_vmax"))
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            if cfg.get("residual"):
                model = ResidualValueUpdater(zero_init=cfg.get("zero_init", True))
            else:
                model = RecurrentValueUpdater()
            loss_fn = make_loss(cfg.get("op_vmax"), cfg.get("residual", False), seed)
            info = train_model(f"{name}_s{seed}", model, loss_fn, train, val, targs, device,
                               seed=seed, ckpt_dir=CKPT_SUBDIR, val_loss_fn=recurrent_loss, verbose=False)
            ckpt = torch.load(os.path.join(CKPT_SUBDIR, f"{name}_s{seed}.pt"), map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            run = diagnose_model(model, test, base_max, device)
            run.update(seed=seed, coverage=coverage, epochs_run=info["epochs_run"],
                       best_epoch=info["best_epoch"], best_val_loss=info["best_val_loss"])
            runs.append(run)
            print(f"{name:18s} seed {seed}: coverage {coverage:.2f}  fixed-pt err "
                  f"{run['median_fixed_point_error_sup']:.4f}  true resid "
                  f"{run['median_true_bellman_residual_final']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        summary["variants"][name] = {"config": cfg, "aggregate": aggregate(runs), "runs": runs}

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    write_report(summary, os.path.join(OUT_DIR, "report.md"))
    plot(summary, exact_curve)
    print(f"done in {time.time() - t_all:.0f}s")
    print(open(os.path.join(OUT_DIR, "report.md")).read())


if __name__ == "__main__":
    main()
