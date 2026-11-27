# Fixed-point diagnostics of the learned recurrent updater

## 1–2. Long run and Bellman residuals (medians over 1,000 test MDPs)

| iter | learned value error | exact VI value error | true residual ‖TV̂−V̂‖∞ | self residual ‖T̂V̂−V̂‖∞ | gap to exact trajectory ‖V̂_t−V_t‖∞ |
|---:|---:|---:|---:|---:|---:|
| 16 | 1.3 | 1.28 | 0.16 | 0.15 | 0.0322 |
| 32 | 0.2839 | 0.238 | 0.041 | 0.0214 | 0.0651 |
| 64 | 0.1023 | 0.00818 | 0.0225 | 0.000506 | 0.119 |
| 96 | 0.09767 | 0.000281 | 0.0221 | 1.24e-05 | 0.123 |
| 128 | 0.09752 | 9.58e-06 | 0.0221 | 4.77e-07 | 0.123 |
| 256 | 0.09752 | 6.14e-08 | 0.0221 | 0 | 0.123 |
| 512 | 0.09752 | 6.14e-08 | 0.0221 | 0 | 0.123 |
| 1024 | 0.09752 | 6.14e-08 | 0.0221 | 0 | 0.123 |

Error rises from 96 to 1,024 on 0.9% of MDPs.

## 3. Fixed-point bias (V̂₁₀₂₄ − V*)

- Mean signed bias -0.1939, mean absolute bias 0.1952; 1.6% of states are overestimated.
- 96.5% of MDPs have the same bias sign on all 8 states; a per-MDP constant offset explains 96.9% of the squared bias.

| V* bin | states | mean bias |
|---|---:|---:|
| [-8, -7) | 1 | +2.7545 |
| [-1, 0) | 3 | +0.0872 |
| [0, 1) | 13 | -0.0067 |
| [1, 2) | 82 | -0.0050 |
| [2, 3) | 203 | -0.0191 |
| [3, 4) | 529 | -0.0422 |
| [4, 5) | 926 | -0.0410 |
| [5, 6) | 1501 | -0.0516 |
| [6, 7) | 1700 | -0.1245 |
| [7, 8) | 1610 | -0.2827 |
| [8, 9) | 1064 | -0.4765 |
| [9, 10) | 368 | -0.6487 |

### Training coverage

- Largest successor value the MLP saw in training (V_0..V_15): 7.94 (99th percentile 6.87).
- 19.0% of test V* entries, and 35.9% of test MDPs, go above that.
- Median fixed-point error ‖V̂∞ − V*‖∞: 0.0550 for MDPs that stay inside the training range vs 0.4879 for MDPs that leave it.
- f_θ − (r + 0.9v) over the V* range: mean |error| 0.0095 where training covered v, 0.0515 above the training max (mean signed -0.0515). Effective γ (∂f/∂v): 0.896 seen vs 0.879 unseen.

## 4. One-step operator error ‖T̂(V) − T(V)‖∞

| input V | median | p90 |
|---|---:|---:|
| exact V_0 | 0.007951 | 0.02005 |
| exact V_4 | 0.004583 | 0.009001 |
| exact V_16 | 0.01043 | 0.0352 |
| exact V_32 | 0.0203 | 0.07201 |
| exact V_64 | 0.02409 | 0.08305 |
| V* | 0.02423 | 0.08347 |
| learned V_hat_16 | 0.01036 | 0.03373 |
| learned V_hat_64 | 0.02206 | 0.06353 |
| learned V_hat_1024 | 0.02213 | 0.06358 |

Contraction factor of T̂ near V*: median 0.894, max 0.920 (100.0% of MDPs below 1). Late-iteration convergence rate at t=60: median 0.890.

Fixed-point error ‖V̂∞ − V*‖∞: median 0.1234; bound ‖T̂V* − V*‖∞ / (1 − L): median 0.2234; bound holds on 99.3% of MDPs.
