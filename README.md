# Anytime Refinement of Recurrent Value Estimates

A small, reproducible PyTorch experiment that asks:

> Does a learned recurrent value updater keep reducing value error and action regret when it runs for more iterations than it was trained on, or does it plateau, oscillate, or degrade?

The model is a **learned VIN-style value updater**, inspired by Value Iteration Networks and recurrent value iteration. It is not a new architecture. The full experiment specification is in [docs/SPEC.md](docs/SPEC.md).

## Result summary

On 1,000 unseen test MDPs, the updater kept improving well past its 16-iteration training horizon. It then settled at a **biased fixed point**, while exact value iteration kept converging. Under the pre-registered criteria the experiment is **not** a positive result: median action regret is already 0 at iteration 16, so it cannot decrease.

| iter | median value error | mean value error | mean regret | % optimal actions | exact VI median error |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.952 | 5.817 | 0.0752 | 80.55 | 5.95 |
| 4 | 4.430 | 4.358 | 0.0167 | 92.29 | 4.42 |
| 8 | 2.985 | 2.896 | 0.00384 | 96.78 | 2.98 |
| **16** (train horizon) | **1.300** | 1.280 | 0.000667 | 98.59 | 1.28 |
| 24 | 0.575 | 0.615 | **0.000206** | **99.26** | 0.552 |
| 32 | 0.284 | 0.356 | 0.000262 | 99.26 | 0.238 |
| 48 | 0.126 | 0.219 | 0.000379 | 99.21 | 0.0442 |
| 64 | 0.102 | 0.199 | 0.000380 | 99.20 | 0.00818 |
| 96 | 0.0977 | 0.195 | 0.000420 | 99.19 | 0.000281 |
| one-shot MLP | 1.043 | 1.224 | 0.107 | 75.00 | – |

Median regret is 0.038 at iteration 1, 0.012 at iteration 2, and exactly 0 from iteration 4 onward.

- **Value error kept falling beyond the training horizon.** The median fell from 1.30 at iteration 16 to 0.102 at iteration 64, a 92% reduction. Value error improved for 100% of test MDPs from 16→32, 100% from 32→64, and 99.1% from 64→96.
- **Value error then plateaued.** From 64→96 the median fell only about 4.5%, levelling off near 0.098. Exact VI kept shrinking geometrically, to 2.8e-4 at iteration 96. The learned operator converges to a stable fixed point that is not V\*. That is expected, because nothing forces f_θ to be exactly `r + 0.9·v`.
- **Action quality peaked shortly after the horizon.** Mean regret fell from 6.7e-4 at iteration 16 to 2.1e-4 at iteration 24. It then crept back up to 4.2e-4 by iteration 96. Even so, 16→64 is still lower overall (6.7e-4 → 3.8e-4). The share of optimal actions stayed at about 99.2%.
- **No collapse, no divergence, no oscillation.** The spread of values across states at iteration 96 (std 0.52) matches that of V\* (0.53).
- **The one-shot MLP is much worse than iterating.** Its median error is 1.04 and its mean regret is 0.107. The recurrent updater passes it in value error at iteration 19, and in regret by iteration 1.

**Outcome label:** continued refinement, then a fixed-point plateau.

Success criteria, measured from iteration 16 to 64:

| Criterion | Result |
|---|---|
| (1) Median value error decreases | ✅ 1.30 → 0.102 |
| (2) Median action regret decreases | ❌ 0 → 0, a floor effect; mean regret does fall, 6.7e-4 → 3.8e-4 |
| (3) Tested on unseen MDPs | ✅ |
| (4) No collapse or divergence | ✅ |

The full report is in [results/report.md](results/report.md). Every metric at every iteration from 1 to 96 is in [results/per_iteration.csv](results/per_iteration.csv) and `results/summary.json`.

### Plots (`results/plots/`)

1. [Median value error vs iteration (log scale)](results/plots/1_value_error.png)
2. [Action regret vs iteration](results/plots/2_action_regret.png)
3. [% of MDPs improving per interval](results/plots/3_pct_improving.png)
4. [Exact VI error vs iteration](results/plots/4_exact_vi_error.png)
5. [Learned vs exact VI error](results/plots/5_learned_vs_exact.png)

## Setup

This needs Python 3.10–3.13, since PyTorch wheels may not exist yet for newer Python versions.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m src.train      # trains both models, ~30 s on an M-series Mac
```

```bash
python -m src.evaluate   # 1,000 test MDPs, 96 iterations
```

```bash
python -m src.plots      # writes results/plots/*.png
```

```bash
python -m src.diagnose   # fixed-point diagnostics, 1,024 iterations
```

```bash
python -m src.coverage   # training-coverage experiment, 7 variants x 3 seeds, ~14 min on CPU
```

```bash
python -m src.coverage --ci   # bootstrap confidence intervals from the saved checkpoints
```

```bash
python -m src.residual   # structure-preserving updaters, ~15 min of training + ~15 min of evaluation on CPU
```

```bash
python -m pytest -q      # 30 unit tests
```

`python -m src.train --small` is the fallback configuration from the spec: 5,000 training MDPs and 50 epochs. It was not needed, because the default run takes about 25 s in total. `--device cpu|mps|auto` picks the device; the default is MPS when available, otherwise CPU.

## Expected outputs

| Path | Contents |
|---|---|
| `checkpoints/recurrent.pt`, `checkpoints/oneshot.pt` | Best-validation weights (gitignored) |
| `results/train_summary.json`, `results/train.log` | Config, per-epoch losses, best epoch |
| `results/summary.json` | Per-iteration mean and median of every metric, criteria, outcome label |
| `results/report.md` | Readable results table |
| `results/per_iteration.csv` | Every metric at every iteration 1..96, for the learned model and exact VI |
| `results/metrics.npz` | Raw per-MDP metrics and all predicted value vectors (gitignored, 6 MB) |
| `results/plots/*.png` | The five required plots |
| `results/diagnostics/` | Fixed-point diagnostics: `report.md`, `summary.json`, three plots |
| `results/coverage/` | Training-coverage experiment: `report.md`, `summary.json`, `ci.md` (bootstrap CIs), three plots |
| `results/residual/` | Structure-preserving updaters: `report.md`, `summary.json`, two plots |

## What is implemented

- **Environment** ([src/mdp.py](src/mdp.py)): 8 states, 3 actions, deterministic transitions to uniformly random successors, rewards drawn from U[-1, 1], and γ = 0.9. There are no terminal states. The train, validation, and test splits use fixed, distinct seeds (1000, 2000, and 3000).
- **Exact reference** ([src/mdp.py](src/mdp.py)): NumPy value iteration in float64, starting from V₀ = 0. It stops when max |ΔV| < 1e-8 or after 500 iterations, and stores the full trajectory. Test MDPs converge in 143–176 iterations. This code does not depend on torch.
- **Learned recurrent updater** ([src/models.py](src/models.py)): one shared MLP `2→64→GELU→64→GELU→1` maps `[r(s,a), V_t(next(s,a))]` to `Q_t(s,a)`, and `V_{t+1}(s) = max_a Q_t(s,a)`. The same parameters are used for every state, action, and iteration. The model has 4,417 parameters.
- **One-shot baseline** ([src/models.py](src/models.py)): an MLP `216→256→GELU→256→GELU→8`. Its input is a one-hot encoding of `next_state` (8·3·8 values) concatenated with the flattened rewards (24 values).
- **Training** ([src/train.py](src/train.py)): the recurrent loss is the MSE against exact V₁…V₁₆, averaged over iterations, states, and MDPs. The one-shot loss is the MSE against V\*. Both use 10k training and 1k validation MDPs, batch size 128, Adam at lr 1e-3, at most 100 epochs, early-stopping patience 10, gradient-norm clipping at 1.0, and a reshuffle every epoch. The best-validation checkpoint is saved. Seed 42 is set for Python `random`, NumPy, and torch (CPU and MPS).
  - The recurrent model stopped early at epoch 26; its best epoch was 16, with validation MSE 3.3e-4.
  - The one-shot model overfits quickly; its best epoch was 4, with validation MSE 2.36.
- **Evaluation** ([src/evaluate.py](src/evaluate.py)): the learned model's value vectors are recorded at every iteration from 1 to 96. The metrics are value error, max value error, regret of the greedy action under the true Q\*, and the share of states where the greedy action is optimal (ties count as optimal). The script also reports the share of MDPs that improve over 16→32, 32→64, and 64→96, and an outcome classifier that labels the run refinement, plateau, oscillation, or divergence.

## Hypothesis and interpretation

The hypothesis was that an updater trained only on iterations 1–16 would keep refining its estimate when run longer. Three outcomes were possible:

- continued refinement
- fixed-point plateau
- instability

What happened was a mix of the first two. For about 30–40 iterations past the horizon, the learned operator follows the exact VI trajectory closely. It has learned a good approximation of the Bellman backup over the range of values it saw in training, so extra compute keeps paying off. Close to V\*, the small error in f_θ starts to dominate. The learned map is still contractive, so it converges stably, but to a fixed point about 0.1 away from V\* (median). Exact VI does not have this floor.

Regret tells a slightly different story from value error. The greedy policy is already optimal in 98.6% of states at iteration 16 and peaks near iteration 24. After that, the learned fixed point's bias reorders a few near-tied actions, and regret rises a little. So **more inference compute helped value accuracy substantially but did not monotonically improve decisions.**

## Fixed-point diagnostics

`python -m src.diagnose` looks at why the learned updater stops at about 0.1 error. Notation: T is the exact Bellman operator and T̂ is one step of the learned model. Full numbers are in [results/diagnostics/report.md](results/diagnostics/report.md).

1. **Long runs: the plateau is final.** Median value error is 0.0977 at 96 and 0.0975 at 128, 256, 512 and 1,024 iterations. The learned iteration reaches an exact float32 fixed point by about iteration 130. Error rises from 96 to 1,024 on only 0.9% of MDPs, with no late divergence.
2. **Bellman residual: converged, but not to V\*.** The learned operator's own residual ‖T̂V̂−V̂‖∞ falls to 0. The true residual ‖TV̂−V̂‖∞ stays at 0.022 (median). This confirms the central limitation: the model stops at its own fixed point, not at V\*.
3. **The bias is systematic and comes from training coverage.** 98.4% of states are underestimated, and 96.5% of MDPs have the same bias sign on all 8 states. The bias is near zero for V\* below 6 and grows steadily with V\*: mean −0.12 for V\* in [6, 7), −0.48 in [8, 9) and −0.65 in [9, 10).
   - Training only ever shows the MLP successor values from V₀ to V₁₅, and the largest of those is 7.94.
   - At the fixed point, 19% of V\* entries are above 7.94, and 36% of test MDPs have at least one such state.
   - In the range training covered, f_θ is within 0.0095 of r + 0.9v on average and its effective discount ∂f/∂v is 0.896. Above 7.94 it underestimates by 0.052 on average and its effective discount drops to 0.879.
   - Median fixed-point error is 0.055 for MDPs that stay inside the training range and 0.488 for MDPs that leave it.
   - The updater generalises past its training horizon in *iterations*, but not past its training range in *values*. Running longer means visiting larger values than training ever showed it.
4. **Single learned step vs iterating it.** Given the same input V, the one-step error ‖T̂V − TV‖∞ is small: 0.005–0.010 on early iterates and 0.024 at V\* (medians). Iterating amplifies it.
   - T̂ is a contraction near V\*: its estimated Lipschitz factor L has median 0.894 and max 0.920, and the measured convergence rate is 0.890.
   - So the fixed-point error should be at most about ‖T̂V\* − V\*‖∞ / (1 − L). That bound has median 0.22 and holds on 99.3% of MDPs. The rest are probably because L is estimated from random samples, which can miss the worst case.
   - The actual fixed-point error has median 0.123, about 5× the one-step error.
   - The gap between the learned and exact trajectories, ‖V̂_t − V_t‖∞, grows from 0.03 at t = 16 to 0.12 at t ≥ 96.

In short, the updater learned the direction of value iteration and converges stably, but to a biased fixed point. The diagnostics add the reason: most of the bias comes from f_θ extrapolating to successor values above anything in its training trajectories. A contraction factor near 0.9 amplifies that small per-step error by up to about 10×.

Directions this points to (the next section tests broader coverage, plus a residual parameterisation of f):
- train on longer trajectories, or on V\* itself, so the value range is covered
- add a fixed-point or Bellman-residual loss on T̂(V\*) − V\*
- normalise value inputs so that high values are not extrapolation

## Training-coverage experiment

`python -m src.coverage` tests whether the fixed-point bias comes from extrapolation. Each variant keeps the test MDPs, the 2-64-64-1 update MLP and the training configuration, and runs with three seeds (42–44). Only the training data, or how f is set up, changes:

| Variant | What changes | Coverage (largest successor value trained on) |
|---|---|---:|
| `baseline` | V₁…V₁₆ trajectories, as before | 7.94 |
| `broad_ops_9/10/12` | adds a one-step loss on random value vectors V ~ U[−a, a]⁸, with target T(V) | 9 / 10 / 12 |
| `reward_scale_1.3` | rewards ×1.3 in half the training MDPs, so trajectories reach higher values | 10.32 |
| `residual` | f = r + 0.9v + ε(r, v): ε starts at exactly 0, with an L2 penalty on ε | 7.94 |
| `residual_randinit` | same as `residual`, but ε starts as a random MLP | 7.94 |

Results are the mean ± std over seeds. High-V\* MDPs are the 35.9% of test MDPs whose V\* exceeds 7.94 in some state. The full table is in [results/coverage/report.md](results/coverage/report.md).

| | baseline | broad_ops_9 | broad_ops_10 | broad_ops_12 | reward_scale_1.3 | residual_randinit |
|---|---:|---:|---:|---:|---:|---:|
| fixed-point error ‖V̂∞−V\*‖∞, all MDPs | 0.087 ± 0.028 | 0.054 ± 0.008 | 0.101 ± 0.061 | 0.094 ± 0.044 | 0.059 ± 0.016 | **0.0060 ± 0.0027** |
| … high-V\* MDPs | 0.344 ± 0.092 | 0.291 ± 0.042 | 0.351 ± 0.143 | 0.163 ± 0.025 | 0.085 ± 0.021 | **0.018 ± 0.012** |
| one-step error at V\*, high-V\* MDPs | 0.046 ± 0.012 | 0.036 ± 0.005 | 0.044 ± 0.019 | 0.025 ± 0.003 | 0.019 ± 0.007 | **0.0029 ± 0.0009** |
| \|f − (r + 0.9v)\| for v > 7.94 | 0.023 ± 0.008 | 0.014 ± 0.002 | 0.015 ± 0.008 | 0.013 ± 0.004 | 0.015 ± 0.006 | **0.0027 ± 0.0014** |
| true Bellman residual at 1,024 | 0.017 ± 0.004 | 0.012 ± 0.003 | 0.017 ± 0.009 | 0.018 ± 0.006 | 0.013 ± 0.004 | **0.0013 ± 0.0005** |
| mean regret at 1,024 | 3.4e-4 | 1.9e-4 | 2.4e-4 | 8.5e-5 | 6.2e-5 | **2.1e-6** |

The zero-initialised `residual` variant reaches V\* to within 4e-6, with zero regret. That result is trivial: it starts as exact value iteration and its targets give it no reason to move, so it is not shown in the table.

What this shows:

1. **Coverage helps where it should, but how you add it matters.**
   - Realistic high-value trajectories (`reward_scale_1.3`) cut the fixed-point error on high-V\* MDPs about 4×, from 0.34 to 0.085. They also cut the one-step error at V\* from 0.046 to 0.019 and mean regret from 3.4e-4 to 6.2e-5.
   - Random operator samples help much less, and inconsistently. All three ranges reduce the f error above 7.94 (0.023 → ~0.014), but the fixed-point error barely moves: only `broad_ops_12` improves the high-V\* MDPs clearly, to 0.16. A likely reason is that uniform random value vectors do not look like real iterates, so training effort goes to inputs the recurrence never visits.
   - Early stopping uses only the trajectory validation loss, so these variants stop after about 20 epochs. They may be under-trained on the operator loss.
2. **The plateau is an approximation problem that can be fixed, not a limit of recurrent refinement.** No variant changes the architecture or the recurrence. Better coverage alone moves the learned fixed point closer to V\*. The residual setup moves it about 15× closer (0.087 → 0.006) while training on the same narrow 0–7.94 range.
3. **The residual parameterisation matters more than coverage.**
   - With f = r + 0.9v + ε, the network does not have to rediscover the discount or extrapolate it. Even starting from a random ε, training shrinks ε everywhere, including above the training range. It does so less well there (0.0027 above 7.94 vs 0.0004 below), but the leftover bias is tiny.
   - This supports the idea that learning only the deviation from a known Bellman structure is what makes the operator extrapolate. It also bakes in γ, so it tests far less of what is "learned".
4. **Seed variance is large.** The baseline fixed-point error ranges from 0.062 to 0.126 across seeds. The single MPS run analysed above (0.123) falls inside that range. Differences smaller than about 2× between the non-residual variants should not be over-read.

Bootstrap 95% CIs over test MDPs are in [results/coverage/ci.md](results/coverage/ci.md). They come from 2,000 resamples, with each MDP's error averaged over seeds, and the ratios are paired against the baseline on the same MDPs.
- High-V\* error, relative to baseline: `reward_scale_1.3` 0.21× [0.195, 0.225], `broad_ops_12` 0.47× [0.45, 0.50], `residual_randinit` 0.047× [0.045, 0.052].
- Overall error, relative to baseline: `residual_randinit` 0.085× [0.083, 0.090], and `broad_ops_10` is *worse* at 1.10× [1.05, 1.16].
- These intervals cover only sampling of MDPs. They are much narrower than the spread across seeds, so seed-to-seed variation is the larger source of uncertainty.

The anytime-refinement result stands: every variant keeps improving well past iteration 16 and converges stably. Training-range coverage and the parameterisation of the update set how close the converged answer gets to V\*.

## Structure-preserving updaters

`python -m src.residual` follows up on the residual result. Full tables are in [results/residual/report.md](results/residual/report.md).

The zero-initialised `residual` model above is effectively the "correction disabled" sanity check. It starts as exact value iteration, its targets are exact value iteration, and its ε stays near 1e-6. So its 3.6e-6 error is the float32 precision floor (running the base operator with ε removed gives 3.7e-6), not something it learned. The informative models are the ones whose correction starts random.

**A. What does ε learn when the base is already exact?** Nothing useful. The ideal correction is 0, so any nonzero ε is error.
- In `residual_randinit`, |ε| is about 5e-4 at the inputs the converged iteration actually visits.
- Above the training range (v > 7.94), the mean |ε| is 3–18× larger than inside it, depending on the seed.
- With ε switched off, the same model is exact value iteration.

**B. Discount shift and a misspecified base.** All five variants below train on 16-step exact trajectories, with ε starting from random init. They are tested at γ ∈ {0.80, 0.85, 0.90, 0.95, 0.97}; 0.95 and 0.97 are outside every training range. The table shows the median fixed-point error ‖V̂∞ − V\*‖∞, as mean ± std over 3 seeds.

| variant | update | trained at | γ=0.80 | γ=0.90 | γ=0.95 | γ=0.97 |
|---|---|---|---:|---:|---:|---:|
| `mlp_g0.9` | f(r, v) | γ = 0.9 | 3.56 | 0.115 | 7.2 | 16.7 |
| `mlp_gcond` | f(r, v, γ) | γ ~ U[0.80, 0.92] | 0.119 | 0.313 | 6.0 | 15.1 |
| `res_g0.9` | r + γv + ε(r, v) | γ = 0.9 | **7.0e-4** | **0.0060** | **0.54** | 3.45 ± 1.08 |
| `res_gcond` | r + γv + ε(r, v, γ) | γ ~ U[0.80, 0.92] | 5.6e-4 | 0.0084 | 0.83 | 3.46 ± 0.86 (1 MDP diverged in 1 of 3 seeds) |
| `res_misspec` | r + 0.8v + ε(r, v) | γ = 0.9 | 3.57 | 0.0135 | 7.17 | 16.7 |
| any residual model, ε disabled | exact VI at the supplied γ | – | 3.9e-7 | 3.7e-6 | 1.1e-5 | 1.5e-5 |
| `res_misspec`, ε disabled | VI with discount 0.8 | – | 3.9e-7 | 3.58 | 10.7 | 20.3 |

What this shows:

1. **Feeding γ to the base is what makes a model generalise across discounts, not giving γ to the network.**
   - `res_g0.9` never saw any γ but 0.9, yet it beats both plain MLPs at every test γ, by 4× (γ = 0.97) up to 5,000×, including discounts it was never trained on (7e-4 vs 3.56 at γ = 0.8).
   - Giving the network γ as an input did not help. `mlp_gcond` improves at γ = 0.8 but gets worse at 0.9 and still fails outside the training range. `res_gcond` is no better than `res_g0.9` anywhere, and it has the only divergence in the study.
2. **A learned correction can repair a wrong base.** In `res_misspec` the base uses discount 0.8 while the data uses 0.9.
   - ε learns the missing 0.1·v almost exactly: mean error 0.001 inside the training range and 0.008 above it. It even extrapolates nearly linearly to v = 12 (see [plot](results/residual/2_learned_correction.png)).
   - That cuts the error at γ = 0.9 from 3.58 with the base alone to 0.0135.
   - Because ε cannot see γ, the repair only holds at the γ it was trained on. At other discounts `res_misspec` is no better than the plain MLP.
   - This is the setting the paste asked for: a learned residual improving on an incomplete base. It does, but only at the training discount.
3. **Outside the training discounts, the correction makes things worse.** With an exact base, error at γ = 0.95/0.97 grows to 0.54/3.45, against 1e-5 for the base alone. Relative to the largest possible |V| = 1/(1−γ), that is 2.7% and 10%.
   - Those discounts produce values well outside training (V\* up to about 33). There the correction is extrapolating, the same failure as before, just smaller.
   - The single divergence makes the point sharply. One test MDP has a self-looping state with all-negative rewards, so V\* = −24.6, while training never shows a value below −3.91. That far out, `res_gcond`'s update grows values instead of shrinking them, and the value runs off to −∞.
4. **Regret follows the same pattern.** Residual models with the correct base have regret of about 1e-9 to 2e-6 inside the training discounts and 1e-4 to 3e-4 outside them. The plain MLPs sit between 3e-4 and 2e-2.

The claims of this study, kept separate:
- **Learned recurrent refinement.** A shared learned update trained on 16 steps keeps improving long past that horizon on unseen MDPs, and converges stably. Its fixed point is biased by how well the learned update extrapolates in *value*, and broader value coverage reduces that bias.
- **Structure-preserving refinement.** Making the known Bellman form explicit (r + γv, with γ supplied) cuts the fixed-point error by about 15–20× at the training discount, even with a randomly initialised correction. It also carries over to new discounts, and lets a learned correction fix a misspecified base. Its remaining weak point is the same as the plain model's: the learned part can only be trusted on values and discounts it was trained on, and far outside them it can break contraction.

## Limitations

- The MDPs are tiny (8 states, 3 actions, deterministic) and share one γ. Nothing here says how the result scales to larger or stochastic MDPs, longer effective horizons, or learned transition models.
- The structural prior is strong. The successor lookup and the max over actions are given, so f_θ only has to learn the scalar map (r, v) ↦ r + 0.9v. This is a test of whether a learned update can be iterated stably, not of learning to plan from scratch.
- Median regret has a floor effect. The median MDP has zero regret from iteration 4 onward, so criterion 2 cannot detect improvement after the horizon. Mean regret and % optimal actions are more informative here.
- The main experiment and the diagnostics use one seed. The coverage experiment uses three, and shows the plateau level varies by about 2× across seeds.
- The one-shot baseline uses a particular input encoding (one-hot transitions) and the same training budget. It overfits quickly, and a better-regularized direct predictor might do better.
- None of this suggests recurrent value updaters can replace MCTS or other search. It is a controlled first test of whether a learned recurrent value update supports useful inference-time computation.
