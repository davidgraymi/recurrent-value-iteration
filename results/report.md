# Results: anytime refinement of a learned recurrent value updater

Test set: 1000 unseen MDPs. Training horizon: 16 iterations.

**Outcome: continued refinement, then fixed-point plateau.** Positive result for anytime refinement: **no**.

## Selected iterations

| iter | median err | mean err | median max-err | median regret | mean regret | % optimal actions | exact VI median err |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.952 | 5.817 | 6.495 | 0.03799 | 0.07515 | 80.55 | 5.95 |
| 2 | 5.405 | 5.304 | 5.844 | 0.01203 | 0.04154 | 86.49 | 5.4 |
| 4 | 4.43 | 4.358 | 4.706 | 0 | 0.01672 | 92.29 | 4.42 |
| 8 | 2.985 | 2.896 | 3.101 | 0 | 0.003835 | 96.78 | 2.98 |
| 16 | 1.3 | 1.28 | 1.348 | 0 | 0.0006665 | 98.59 | 1.28 |
| 24 | 0.5753 | 0.6146 | 0.6024 | 0 | 0.0002063 | 99.26 | 0.552 |
| 32 | 0.2839 | 0.3563 | 0.307 | 0 | 0.0002623 | 99.26 | 0.238 |
| 48 | 0.126 | 0.2193 | 0.1502 | 0 | 0.0003794 | 99.21 | 0.0442 |
| 64 | 0.1023 | 0.1989 | 0.1273 | 0 | 0.0003795 | 99.20 | 0.00818 |
| 96 | 0.09767 | 0.1953 | 0.1235 | 0 | 0.0004199 | 99.19 | 0.000281 |
| one-shot | 1.043 | 1.224 | 1.754 | 0.07049 | 0.1071 | 75.00 | – |

## % of test MDPs whose value error improves

- iterations 16->32: 100.0%
- iterations 32->64: 100.0%
- iterations 64->96: 99.1%

## Success criteria (16 → 64)

1. Median value error decreases: **True** (1.3 → 0.1023, -92.1%)
2. Median action regret decreases: **False** (0 → 0); mean regret 0.0006665 → 0.0003795
3. Evaluated on unseen MDPs: **True** (independent seed for test split)
4. No collapse / divergence: **True** (state-value std at 96: 0.522 vs V* std 0.534; median error at 96 = 0.09767)
