# Structure-preserving updaters

## A. The learned correction ε(r, v) (coverage-experiment checkpoints)

Ideal correction here is ε = 0, because the base r + 0.9v is already the exact Bellman map: any nonzero ε is error, not a learned improvement.

| model | seed | mean \|ε\|, v ≤ 7.94 | mean \|ε\|, v ∈ (7.94, 10] | max \|ε\|, v ∈ (7.94, 10] | median \|ε\| on fixed-point inputs | fixed-pt error with ε disabled |
|---|---:|---:|---:|---:|---:|---:|
| residual | 42 | 2.6e-07 | 8.6e-07 | 2.5e-06 | 1.8e-07 | 3.7e-06 |
| residual | 43 | 6.9e-07 | 3.1e-06 | 1.1e-05 | 4.4e-07 | 3.7e-06 |
| residual | 44 | 3.8e-07 | 1.5e-06 | 5.2e-06 | 5.2e-07 | 3.7e-06 |
| residual_randinit | 42 | 0.00054 | 0.0022 | 0.011 | 0.00051 | 3.7e-06 |
| residual_randinit | 43 | 0.00041 | 0.0013 | 0.0035 | 0.00053 | 3.7e-06 |
| residual_randinit | 44 | 0.00025 | 0.0046 | 0.0085 | 0.00022 | 3.7e-06 |

## B. Discount shift and misspecified base

Training γ: 0.9 for `*_g0.9` and `res_misspec`; U[0.8, 0.92] for `*_gcond`. Test γ = 0.95 and 0.97 are outside every training range. Mean ± std over seeds [42, 43, 44] of the median over 1000 test MDPs; 2048 iterations. MDPs whose values diverge to ±inf are counted in the last table and excluded from the others.

### Median fixed-point error ‖V̂∞ − V*‖∞

| variant | γ=0.8 | γ=0.85 | γ=0.9 | γ=0.95 | γ=0.97 |
|---|---:|---:|---:|---:|---:|
| mlp_g0.9 | 3.56 ± 0.0995 | 2.37 ± 0.1 | 0.115 ± 0.039 | 7.2 ± 0.0965 | 16.7 ± 0.0978 |
| mlp_gcond | 0.119 ± 0.0243 | 0.452 ± 0.145 | 0.313 ± 0.114 | 6 ± 0.133 | 15.1 ± 0.0791 |
| res_g0.9 | 0.000703 ± 0.000192 | 0.000789 ± 0.000251 | 0.00598 ± 0.00267 | 0.537 ± 0.0437 | 3.45 ± 1.08 |
| res_g0.9, ε disabled | 3.86e-07 | 7.11e-07 | 3.69e-06 | 1.13e-05 | 1.5e-05 |
| res_gcond | 0.000556 ± 0.000137 | 0.00113 ± 0.000568 | 0.00843 ± 0.00312 | 0.833 ± 0.192 | 3.46 ± 0.861 |
| res_gcond, ε disabled | 3.86e-07 | 7.11e-07 | 3.69e-06 | 1.13e-05 | 1.5e-05 |
| res_misspec | 3.57 ± 0.0106 | 2.38 ± 0.0107 | 0.0135 ± 0.0077 | 7.17 ± 0.0143 | 16.7 ± 0.00906 |
| res_misspec, ε disabled | 3.86e-07 | 1.2 | 3.58 | 10.7 | 20.3 |

### Same, × (1 − γ) (relative to the largest possible |V|)

| variant | γ=0.8 | γ=0.85 | γ=0.9 | γ=0.95 | γ=0.97 |
|---|---:|---:|---:|---:|---:|
| mlp_g0.9 | 0.71 ± 0.02 | 0.36 ± 0.015 | 0.011 ± 0.0039 | 0.36 ± 0.0048 | 0.5 ± 0.0029 |
| mlp_gcond | 0.024 ± 0.0049 | 0.068 ± 0.022 | 0.031 ± 0.011 | 0.3 ± 0.0067 | 0.45 ± 0.0024 |
| res_g0.9 | 0.00014 ± 3.8e-05 | 0.00012 ± 3.8e-05 | 0.0006 ± 0.00027 | 0.027 ± 0.0022 | 0.1 ± 0.032 |
| res_gcond | 0.00011 ± 2.7e-05 | 0.00017 ± 8.5e-05 | 0.00084 ± 0.00031 | 0.042 ± 0.0096 | 0.1 ± 0.026 |
| res_misspec | 0.71 ± 0.0021 | 0.36 ± 0.0016 | 0.0014 ± 0.00077 | 0.36 ± 0.00071 | 0.5 ± 0.00027 |

### Mean regret

| variant | γ=0.8 | γ=0.85 | γ=0.9 | γ=0.95 | γ=0.97 |
|---|---:|---:|---:|---:|---:|
| mlp_g0.9 | 0.0043 ± 0.0001 | 0.0013 ± 0.0001 | 0.00028 ± 5.6e-05 | 0.0035 ± 0.00017 | 0.0077 ± 0.00021 |
| mlp_gcond | 0.0014 ± 0.00059 | 0.0015 ± 0.00065 | 0.0014 ± 0.00026 | 0.0084 ± 0.0015 | 0.015 ± 0.0033 |
| res_g0.9 | 8.9e-10 ± 1.3e-09 | 1.5e-08 ± 2.1e-08 | 2.1e-06 ± 1.1e-06 | 0.0001 ± 7.1e-05 | 0.00032 ± 0.0002 |
| res_gcond | 1.8e-09 ± 1.3e-09 | 0 | 1.8e-06 ± 9.5e-07 | 0.00014 ± 0.00012 | 0.00023 ± 0.00015 |
| res_misspec | 0.0052 ± 4.4e-05 | 0.0018 ± 7.9e-06 | 1.9e-06 ± 1.2e-06 | 0.0026 ± 7.3e-06 | 0.0065 ± 4.3e-05 |

### % optimal actions

| variant | γ=0.8 | γ=0.85 | γ=0.9 | γ=0.95 | γ=0.97 |
|---|---:|---:|---:|---:|---:|
| mlp_g0.9 | 96.98 ± 0.07 | 98.57 ± 0.10 | 99.20 ± 0.11 | 97.32 ± 0.09 | 96.26 ± 0.10 |
| mlp_gcond | 98.44 ± 0.21 | 98.41 ± 0.25 | 97.97 ± 0.14 | 95.62 ± 0.38 | 94.09 ± 0.61 |
| res_g0.9 | 100.00 ± 0.01 | 100.00 ± 0.01 | 99.96 ± 0.00 | 99.70 ± 0.06 | 99.54 ± 0.14 |
| res_gcond | 99.99 ± 0.01 | 100.00 | 99.92 ± 0.03 | 99.54 ± 0.17 | 99.42 ± 0.10 |
| res_misspec | 96.80 ± 0.01 | 98.31 ± 0.02 | 99.93 ± 0.03 | 97.72 ± 0.03 | 96.68 ± 0.02 |

### Self residual at the last iteration (convergence check)

| variant | γ=0.8 | γ=0.85 | γ=0.9 | γ=0.95 | γ=0.97 |
|---|---:|---:|---:|---:|---:|
| mlp_g0.9 | 0 | 0 | 0 | 0 | 0 |
| mlp_gcond | 0 | 0 | 0 | 0 | 0 |
| res_g0.9 | 0 | 0 | 0 | 0 | 0 |
| res_gcond | 0 | 0 | 0 | 0 | 0 |
| res_misspec | 0 | 0 | 0 | 0 | 0 |

### Test MDPs whose values diverged (excluded from the rows above)

| variant | γ=0.8 | γ=0.85 | γ=0.9 | γ=0.95 | γ=0.97 |
|---|---:|---:|---:|---:|---:|
| mlp_g0.9 | 0 | 0 | 0 | 0 | 0 |
| mlp_gcond | 0 | 0 | 0 | 0 | 0 |
| res_g0.9 | 0 | 0 | 0 | 0 | 0 |
| res_gcond | 0 | 0 | 0 | 0 | 0.33 ± 0.47 |
| res_misspec | 0 | 0 | 0 | 0 | 0 |

### The learned correction at γ = 0.9

Ideal ε: 0 for `res_g0.9` and `res_gcond`; 0.1·v for `res_misspec`. "Seen" is v ∈ [0, 7.94], the training range; "unseen" is v ∈ (7.94, 10].

| variant | mean \|ε\| seen | mean \|ε\| unseen | mean \|ε − ideal\| seen | mean \|ε − ideal\| unseen |
|---|---:|---:|---:|---:|
| res_g0.9 | 0.0004 | 0.00273 | 0.0004 | 0.00273 |
| res_gcond | 0.000536 | 0.00516 | 0.000536 | 0.00516 |
| res_misspec | 0.394 | 0.892 | 0.000996 | 0.00798 |
