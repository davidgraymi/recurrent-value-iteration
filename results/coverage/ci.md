# Bootstrap confidence intervals (coverage experiment)

95% percentile bootstrap over test MDPs (2,000 resamples). Each MDP's error is first averaged over seeds [42, 43, 44], so these intervals reflect variation across MDPs, not across seeds. All MDPs: n = 1000; high-V* MDPs: n = 359. Ratios are paired (same MDPs resampled for both variants); a ratio interval that excludes 1 is a difference from baseline beyond MDP-sampling noise.

| variant | median ‖V̂∞−V*‖∞, all | ratio to baseline, all | median ‖V̂∞−V*‖∞, high-V* | ratio to baseline, high-V* | mean regret |
|---|---:|---:|---:|---:|---:|
| baseline | 0.0854 [0.0779, 0.0919] | 1 [1, 1] | 0.343 [0.314, 0.372] | 1 [1, 1] | 0.000335 [0.000215, 0.000483] |
| broad_ops_9 | 0.0523 [0.0491, 0.056] | 0.613 [0.586, 0.642] | 0.291 [0.268, 0.315] | 0.849 [0.84, 0.858] | 0.000193 [0.000107, 0.000296] |
| broad_ops_10 | 0.0943 [0.0832, 0.106] | 1.1 [1.05, 1.16] | 0.351 [0.329, 0.374] | 1.02 [1, 1.05] | 0.000244 [0.000165, 0.000342] |
| broad_ops_12 | 0.0968 [0.0949, 0.1] | 1.13 [1.08, 1.22] | 0.161 [0.155, 0.168] | 0.468 [0.449, 0.499] | 8.52e-05 [5.53e-05, 0.00012] |
| reward_scale_1.3 | 0.066 [0.0649, 0.067] | 0.773 [0.721, 0.842] | 0.0708 [0.0692, 0.0744] | 0.206 [0.195, 0.225] | 6.19e-05 [3.34e-05, 9.46e-05] |
| residual | 3.82e-06 [3.55e-06, 4.09e-06] | 4.48e-05 [4.21e-05, 4.76e-05] | 1.45e-05 [1.22e-05, 1.72e-05] | 4.23e-05 [3.9e-05, 4.8e-05] | 0 [0, 0] |
| residual_randinit | 0.00726 [0.00694, 0.00774] | 0.085 [0.0827, 0.0896] | 0.0163 [0.0142, 0.0187] | 0.0473 [0.045, 0.0517] | 2.07e-06 [3.79e-07, 4.56e-06] |
