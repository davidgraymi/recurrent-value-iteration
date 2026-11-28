"""Structure-preserving updaters: the learned correction, discount shift, and a misspecified base.

Usage:
    python -m src.residual                 # all parts, seeds 42-44 (~15 min on CPU)
    python -m src.residual --seeds 42
    python -m src.residual --eval-only     # re-evaluate saved checkpoints

Part A: the correction eps_theta(r, v) in the coverage experiment's `residual` and
        `residual_randinit` checkpoints, measured directly, plus the "correction
        disabled" check (base operator alone = exact value iteration).
Part B: generalisation to a changed discount, and a misspecified base. Five variants,
        each trained on 16-step exact VI trajectories of the standard training MDPs:

  mlp_g0.9      f(r, v)                       trained at gamma = 0.9
  mlp_gcond     f(r, v, gamma)                trained at gamma ~ U[0.80, 0.92]
  res_g0.9      r + gamma v + eps(r, v)       trained at gamma = 0.9; gamma supplied at test
  res_gcond     r + gamma v + eps(r, v, gamma) trained at gamma ~ U[0.80, 0.92]
  res_misspec   r + 0.8 v + eps(r, v)         trained at gamma = 0.9; base discount is wrong,
                                              so eps must learn the missing 0.1 v

All corrections start from PyTorch's default (random) init, so no variant starts
as exact value iteration. Each is tested on the standard test MDPs at
gamma in {0.80, 0.85, 0.90, 0.95, 0.97}; 0.95 and 0.97 are outside every training
range. Writes results/residual/{summary.json, report.md, *.png}.
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

from src.coverage import CKPT_SUBDIR as COVERAGE_CKPTS  # noqa: E402
from src.diagnose import sup  # noqa: E402
from src.evaluate import N_TEST, compute_metrics  # noqa: E402
from src.mdp import DATA_SEEDS, bellman_optimality, generate_mdps, make_dataset, value_iteration  # noqa: E402
from src.models import ConditionedValueUpdater, ResidualValueUpdater  # noqa: E402
from src.train import CKPT_DIR, RESULTS_DIR, TRAIN_ITERS, train_model  # noqa: E402

OUT_DIR = os.path.join(RESULTS_DIR, "residual")
CKPT_SUBDIR = os.path.join(CKPT_DIR, "residual")
EPS_WEIGHT = 1e-3
TEST_GAMMAS = [0.80, 0.85, 0.90, 0.95, 0.97]
TRAIN_RANGE = (0.80, 0.92)
EVAL_ITERS = 2048
BASE_MAX = 7.94  # largest successor value in the gamma = 0.9 training trajectories
VARIANTS = {
    "mlp_g0.9": dict(gamma_input=False, residual=False, train_gamma=0.9),
    "mlp_gcond": dict(gamma_input=True, residual=False, train_gamma=TRAIN_RANGE),
    "res_g0.9": dict(gamma_input=False, residual=True, train_gamma=0.9),
    "res_gcond": dict(gamma_input=True, residual=True, train_gamma=TRAIN_RANGE),
    "res_misspec": dict(gamma_input=False, residual=True, base_gamma=0.8, train_gamma=0.9),
}
DEFAULT_SEEDS = [42, 43, 44]


# ---------------------------------------------------------------- data


def gamma_dataset(split, n, train_gamma, seed_offset=0):
    """Standard MDPs of `split`, solved with a per-MDP discount.

    train_gamma: a float (same discount for every MDP) or a (lo, hi) range to
    sample uniformly per MDP.
    """
    ns, r = generate_mdps(n, DATA_SEEDS[split])
    if isinstance(train_gamma, tuple):
        rng = np.random.default_rng(DATA_SEEDS[split] + 7 + seed_offset)
        gamma = rng.uniform(*train_gamma, size=n)
    else:
        gamma = np.full(n, float(train_gamma))
    traj, v_star, _ = value_iteration(ns, r, gamma[:, None, None], max_iter=5000)
    return {"next_state": ns, "reward": r, "gamma": gamma, "trajectory": traj, "v_star": v_star}


def to_tensors(d, device):
    return {
        "next_state": torch.as_tensor(d["next_state"], dtype=torch.long, device=device),
        "reward": torch.as_tensor(d["reward"], dtype=torch.float32, device=device),
        "gamma": torch.as_tensor(d["gamma"], dtype=torch.float32, device=device),
        "targets": torch.as_tensor(d["trajectory"][:, 1 : TRAIN_ITERS + 1], dtype=torch.float32, device=device),
    }


def traj_loss(model, batch):
    pred = model(batch["next_state"], batch["reward"], TRAIN_ITERS, batch["gamma"])
    return ((pred - batch["targets"]) ** 2).mean()


def train_loss(model, batch):
    loss = traj_loss(model, batch)
    if model.residual and model.training and model.eps_penalty:
        loss = loss + EPS_WEIGHT * torch.stack(model.eps_penalty).mean()
        model.eps_penalty.clear()
    return loss


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def run_to_convergence(model, ns, r, gamma, device, n_iters=EVAL_ITERS):
    """Iterate `n_iters` times; returns the final V [N, S] and the last self-residual."""
    model.eval().to(device)
    ns_t = torch.as_tensor(ns, device=device)
    r_t = torch.as_tensor(r, dtype=torch.float32, device=device)
    g_t = torch.full((len(r),), gamma, dtype=torch.float32, device=device)
    V = torch.zeros(len(r), ns.shape[1], device=device)
    for _ in range(n_iters):
        V_prev, V = V, model.step(V, ns_t, r_t, g_t)
    return V.cpu().double().numpy(), sup((V - V_prev).cpu().double().numpy())


@torch.no_grad()
def eval_at_gamma(model, test, gamma, device):
    ns, r = test["next_state"], test["reward"]
    v_star = test["v_star"][gamma]
    v_inf, self_res = run_to_convergence(model, ns, r, gamma, device)
    ok = np.isfinite(v_inf).all(axis=1)  # diverged MDPs are reported, then excluded
    ns, r, v_star, v_inf, self_res = ns[ok], r[ok], v_star[ok], v_inf[ok], self_res[ok]
    m = compute_metrics(v_inf[:, None], v_star, ns, r, gamma=gamma)
    fp = sup(v_inf - v_star)
    true_res = sup(bellman_optimality(v_inf, ns, r, gamma) - v_inf)
    return {
        "n_diverged": int((~ok).sum()),
        "median_fp_error": float(np.median(fp)),
        "median_fp_error_scaled": float(np.median(fp) * (1 - gamma)),  # relative to max |V| = 1/(1-gamma)
        "median_true_residual": float(np.median(true_res)),
        "median_self_residual": float(np.median(self_res)),
        "mean_regret": float(m["regret"][:, 0].mean()),
        "pct_optimal": float(m["pct_optimal"][:, 0].mean()),
    }


@torch.no_grad()
def eps_on_grid(model, gamma, r_grid, v_grid, device):
    """eps_theta on an (r, v) grid at a given gamma. Returns [len(r), len(v)]."""
    R, Vg = np.meshgrid(r_grid, v_grid, indexing="ij")
    Rt = torch.as_tensor(R, dtype=torch.float32, device=device)
    Vt = torch.as_tensor(Vg, dtype=torch.float32, device=device)
    return model.correction(Rt, Vt, torch.tensor(gamma, device=device)).cpu().double().numpy()


# ---------------------------------------------------------------- part A


@torch.no_grad()
def part_a(device):
    """Measure eps in the coverage experiment's residual checkpoints."""
    test = make_dataset("test", N_TEST)
    ns, r, v_star = test["next_state"], test["reward"], test["v_star"]
    r_grid = np.linspace(-1, 1, 41)
    v_seen = np.linspace(0, BASE_MAX, 81)
    v_unseen = np.linspace(BASE_MAX, 10, 41)
    out = {}
    for name, zero_init in (("residual", True), ("residual_randinit", False)):
        rows = []
        for seed in DEFAULT_SEEDS:
            path = os.path.join(COVERAGE_CKPTS, f"{name}_s{seed}.pt")
            if not os.path.exists(path):
                continue
            m = ResidualValueUpdater(zero_init=zero_init)
            m.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
            m.eval().to(device)

            def eps(v_values):
                x = torch.as_tensor(np.stack(np.meshgrid(r_grid, v_values, indexing="ij"), -1), dtype=torch.float32)
                return m.f(x.to(device)).squeeze(-1).cpu().double().numpy()

            e_seen, e_unseen = eps(v_seen), eps(v_unseen)
            # eps on the inputs the iteration actually visits at its fixed point
            ns_t = torch.as_tensor(ns, device=device)
            r_t = torch.as_tensor(r, dtype=torch.float32, device=device)
            V = torch.zeros(len(r), 8, device=device)
            for _ in range(1024):
                V = m.step(V, ns_t, r_t)
            succ = torch.gather(V, 1, ns_t.reshape(len(r), -1)).reshape(len(r), 8, 3)
            e_fp = m.f(torch.stack([r_t, succ], -1)).squeeze(-1).abs().cpu().double().numpy()
            # correction disabled = base operator alone
            m_off = ResidualValueUpdater(zero_init=True).eval()
            V0 = torch.zeros(len(r), 8, dtype=torch.float64)
            for _ in range(1024):
                V0 = m_off.step(V0.float(), ns_t.cpu(), r_t.cpu()).double()
            rows.append({
                "seed": seed,
                "mean_abs_eps_seen_v": float(np.abs(e_seen).mean()),
                "max_abs_eps_seen_v": float(np.abs(e_seen).max()),
                "mean_abs_eps_unseen_v": float(np.abs(e_unseen).mean()),
                "max_abs_eps_unseen_v": float(np.abs(e_unseen).max()),
                "mean_signed_eps_unseen_v": float(e_unseen.mean()),
                "median_abs_eps_at_fixed_point_inputs": float(np.median(e_fp)),
                "p99_abs_eps_at_fixed_point_inputs": float(np.percentile(e_fp, 99)),
                "eps_off_median_fp_error": float(np.median(sup(V0.numpy() - v_star))),
            })
        out[name] = rows
    return out


# ---------------------------------------------------------------- part B


def make_model(cfg):
    return ConditionedValueUpdater(gamma_input=cfg["gamma_input"], residual=cfg["residual"],
                                   base_gamma=cfg.get("base_gamma"))


def part_b(seeds, device, eval_only=False, previous=None):
    targs = SimpleNamespace(lr=1e-3, epochs=100, batch_size=128, patience=10)
    test_mdps = gamma_dataset("test", N_TEST, 0.9)
    test = {"next_state": test_mdps["next_state"], "reward": test_mdps["reward"], "v_star": {}}
    for g in TEST_GAMMAS:
        _, v_star, n_it = value_iteration(test["next_state"], test["reward"], g, max_iter=5000)
        assert n_it.max() < 5000
        test["v_star"][g] = v_star

    r_grid = np.linspace(-1, 1, 21)
    v_grid = np.linspace(-2, 12, 141)
    results, eps_curves = {}, {}
    for name, cfg in VARIANTS.items():
        if not eval_only:
            train = to_tensors(gamma_dataset("train", 10_000, cfg["train_gamma"]), device)
            val = to_tensors(gamma_dataset("val", 1_000, cfg["train_gamma"]), device)
        runs = []
        for seed in seeds:
            t0 = time.time()
            model = make_model(cfg)
            if eval_only:  # reuse saved checkpoints and the training record from summary.json
                prev = next(r for r in previous[name]["runs"] if r["seed"] == seed)
                info = {k: prev[k] for k in ("epochs_run", "best_val_loss")}
            else:
                info = train_model(f"{name}_s{seed}", model, train_loss, train, val, targs, device,
                                   seed=seed, ckpt_dir=CKPT_SUBDIR, val_loss_fn=traj_loss, verbose=False)
            model.load_state_dict(torch.load(os.path.join(CKPT_SUBDIR, f"{name}_s{seed}.pt"),
                                             map_location="cpu")["state_dict"])
            run = {"seed": seed, "epochs_run": info["epochs_run"], "best_val_loss": info["best_val_loss"],
                   "by_gamma": {str(g): eval_at_gamma(model, test, g, device) for g in TEST_GAMMAS}}
            if cfg["residual"]:
                model.use_correction = False
                run["eps_off_by_gamma"] = {str(g): eval_at_gamma(model, test, g, device) for g in TEST_GAMMAS}
                model.use_correction = True
                e = eps_on_grid(model, 0.9, r_grid, v_grid, device)
                eps_curves.setdefault(name, []).append(e.mean(axis=0).tolist())
                seen = (v_grid >= 0) & (v_grid <= BASE_MAX)
                unseen = (v_grid > BASE_MAX) & (v_grid <= 10)
                target = 0.1 * v_grid if cfg.get("base_gamma") == 0.8 else 0 * v_grid
                err = e - target[None, :]
                run["eps_at_gamma_0.9"] = {
                    "mean_abs_seen_v": float(np.abs(e[:, seen]).mean()),
                    "mean_abs_unseen_v": float(np.abs(e[:, unseen]).mean()),
                    "mean_abs_error_vs_ideal_seen_v": float(np.abs(err[:, seen]).mean()),
                    "mean_abs_error_vs_ideal_unseen_v": float(np.abs(err[:, unseen]).mean()),
                }
            runs.append(run)
            g9 = run["by_gamma"]["0.9"]["median_fp_error"]
            g97 = run["by_gamma"]["0.97"]["median_fp_error"]
            print(f"{name:12s} seed {seed}: fp err @0.90 {g9:.3g}  @0.97 {g97:.3g}  "
                  f"({info['epochs_run']} epochs, {time.time() - t0:.0f}s)", flush=True)
        results[name] = {"config": {k: v for k, v in cfg.items()}, "runs": runs}
    return results, eps_curves, v_grid


def agg(runs, key, metric):
    vals = np.array([[r[key][str(g)][metric] for g in TEST_GAMMAS] for r in runs])
    return vals.mean(0), vals.std(0)


# ---------------------------------------------------------------- report & plots


def write_report(summary, path):
    A, B = summary["part_a"], summary["part_b"]
    lines = ["# Structure-preserving updaters", "", "## A. The learned correction ε(r, v) (coverage-experiment checkpoints)", "",
             "Ideal correction here is ε = 0, because the base r + 0.9v is already the exact Bellman map: "
             "any nonzero ε is error, not a learned improvement.", "",
             "| model | seed | mean \\|ε\\|, v ≤ 7.94 | mean \\|ε\\|, v ∈ (7.94, 10] | max \\|ε\\|, v ∈ (7.94, 10] | median \\|ε\\| on fixed-point inputs | fixed-pt error with ε disabled |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, rows in A.items():
        for row in rows:
            lines.append(f"| {name} | {row['seed']} | {row['mean_abs_eps_seen_v']:.2g} | {row['mean_abs_eps_unseen_v']:.2g} | "
                         f"{row['max_abs_eps_unseen_v']:.2g} | {row['median_abs_eps_at_fixed_point_inputs']:.2g} | "
                         f"{row['eps_off_median_fp_error']:.2g} |")
    lines += ["", "## B. Discount shift and misspecified base", "",
              f"Training γ: 0.9 for `*_g0.9` and `res_misspec`; U{list(TRAIN_RANGE)} for `*_gcond`. "
              f"Test γ = 0.95 and 0.97 are outside every training range. Mean ± std over seeds {summary['seeds']} "
              f"of the median over {N_TEST} test MDPs; {EVAL_ITERS} iterations. MDPs whose values diverge to ±inf "
              f"are counted in the last table and excluded from the others.", ""]
    for metric, label, fmt in (("median_fp_error", "Median fixed-point error ‖V̂∞ − V*‖∞", "{:.3g}"),
                               ("median_fp_error_scaled", "Same, × (1 − γ) (relative to the largest possible |V|)", "{:.2g}"),
                               ("mean_regret", "Mean regret", "{:.2g}"),
                               ("pct_optimal", "% optimal actions", "{:.2f}"),
                               ("median_self_residual", "Self residual at the last iteration (convergence check)", "{:.1g}"),
                               ("n_diverged", "Test MDPs whose values diverged (excluded from the rows above)", "{:.2g}")):
        lines += [f"### {label}", "", "| variant | " + " | ".join(f"γ={g}" for g in TEST_GAMMAS) + " |",
                  "|---|" + "---:|" * len(TEST_GAMMAS)]
        for name, res in B.items():
            mu, sd = agg(res["runs"], "by_gamma", metric)
            lines.append(f"| {name} | " + " | ".join(fmt.format(m) + (f" ± {fmt.format(s)}" if s > 0 else "") for m, s in zip(mu, sd)) + " |")
            if metric == "median_fp_error" and "eps_off_by_gamma" in res["runs"][0]:
                mu0, _ = agg(res["runs"], "eps_off_by_gamma", metric)
                lines.append(f"| {name}, ε disabled | " + " | ".join(fmt.format(m) for m in mu0) + " |")
        lines.append("")
    lines += ["### The learned correction at γ = 0.9", "",
              "Ideal ε: 0 for `res_g0.9` and `res_gcond`; 0.1·v for `res_misspec`. "
              "\"Seen\" is v ∈ [0, 7.94], the training range; \"unseen\" is v ∈ (7.94, 10].", "",
              "| variant | mean \\|ε\\| seen | mean \\|ε\\| unseen | mean \\|ε − ideal\\| seen | mean \\|ε − ideal\\| unseen |",
              "|---|---:|---:|---:|---:|"]
    for name, res in B.items():
        if "eps_at_gamma_0.9" not in res["runs"][0]:
            continue
        e = {k: np.mean([r["eps_at_gamma_0.9"][k] for r in res["runs"]]) for k in res["runs"][0]["eps_at_gamma_0.9"]}
        lines.append(f"| {name} | {e['mean_abs_seen_v']:.3g} | {e['mean_abs_unseen_v']:.3g} | "
                     f"{e['mean_abs_error_vs_ideal_seen_v']:.3g} | {e['mean_abs_error_vs_ideal_unseen_v']:.3g} |")
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def plot(summary, eps_curves, v_grid):
    B = summary["part_b"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, metric, ylabel in ((axes[0], "median_fp_error", "median ‖V̂∞ − V*‖∞"),
                               (axes[1], "mean_regret", "mean regret")):
        for name, res in B.items():
            mu, sd = agg(res["runs"], "by_gamma", metric)
            ls = "--" if name.startswith("mlp") else "-"
            ax.errorbar(TEST_GAMMAS, np.maximum(mu, 1e-9), yerr=sd, ls=ls, marker="o", ms=4, capsize=2, label=name)
        ax.axvspan(*TRAIN_RANGE, color="gray", alpha=0.1, label="γ-conditioned training range")
        ax.axvline(0.9, color="gray", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("test discount γ")
        ax.set_ylabel(ylabel + " (mean ± std over seeds)")
    axes[0].set_title("Fixed-point error under a discount shift")
    axes[1].set_title("Action regret under a discount shift")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "1_gamma_shift.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, curves in eps_curves.items():
        for i, c in enumerate(curves):
            ax.plot(v_grid, c, color={"res_g0.9": "C2", "res_gcond": "C3", "res_misspec": "C4"}[name],
                    alpha=0.7, label=name if i == 0 else None)
    ax.plot(v_grid, 0.1 * v_grid, "k--", lw=1, label="ideal ε for res_misspec (0.1·v)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(0, BASE_MAX, color="gray", alpha=0.1, label="successor values seen in training")
    ax.set_xlabel("successor value v")
    ax.set_ylabel("learned correction ε(r, v) at γ = 0.9, averaged over r")
    ax.set_title("What the correction learns (one line per seed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "2_learned_correction.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-only", action="store_true", help="re-evaluate saved checkpoints, no training")
    args = p.parse_args()
    device = torch.device(args.device)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_SUBDIR, exist_ok=True)
    t0 = time.time()
    a = part_a(device)
    previous = None
    if args.eval_only:
        with open(os.path.join(OUT_DIR, "summary.json")) as fh:
            previous = json.load(fh)["part_b"]
    b, eps_curves, v_grid = part_b(args.seeds, device, args.eval_only, previous)
    summary = {"seeds": args.seeds, "part_a": a, "part_b": b,
               "eps_curves_gamma_0.9": {"v_grid": v_grid.tolist(), **eps_curves}}
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    write_report(summary, os.path.join(OUT_DIR, "report.md"))
    plot(summary, eps_curves, v_grid)
    print(f"done in {time.time() - t0:.0f}s")
    print(open(os.path.join(OUT_DIR, "report.md")).read())


if __name__ == "__main__":
    main()
