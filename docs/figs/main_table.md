# Main table (seed-mean best ± std / last evaluation)

Each cell: best of the seed-mean test curve ± std at that round / mean of the last evaluation (round, seeds). FedScout rows: scout alone = the shared c8 after FedAvg with its head fine-tuned on local data for one epoch (no private model); fused = scout + alpha * learner (alpha on held-out).

| method | Cifar100 | Flowers102 | TinyImagenet |
|---|---|---|---|
| FedProto | 38.24 ± 0.09 / 38.03 (R500, n=3) | 49.88 ± 1.11 / 49.33 (R500, n=3) | 18.24 ± 0.22 / 18.13 (R500, n=3) |
| FedGH | 39.83 ± 0.26 / 39.77 (R500, n=3) | 53.06 ± 0.66 / 53.01 (R500, n=3) | 23.19 ± 0.13 / 22.36 (R500, n=3) |
| LG-FedAvg | 40.64 ± 0.41 / 40.64 (R500, n=3) | 53.80 ± 0.51 / 52.91 (R500, n=3) | 23.64 ± 0.14 / 23.01 (R500, n=3) |
| FedRG | 41.98 ± 0.11 / 41.69 (R500, n=3) | 50.62 ± 0.39 / 50.62 (R500, n=3) | 24.15 ± 0.15 / 23.85 (R500, n=3) |
| FedTGP | 44.51 ± 0.27 / 44.51 (R500, n=3) | 55.60 ± 0.53 / 55.47 (R500, n=3) | 23.20 ± 0.37 / 23.20 (R500, n=3) |
| FedSA | 44.89 ± 0.29 / 44.79 (R500, n=3) | 55.59 ± 0.19 / 55.33 (R500, n=3) | 23.92 ± 0.05 / 22.79 (R500, n=3) |
| FedMRL (c8) | 38.24 ± 0.11 / 37.77 (R500, n=3) | 48.79 ± 0.68 / 48.55 (R500, n=3) | — |
| pFedMoE (c8) | 40.77 ± 0.43 / 40.45 (R500, n=3) | — | — |
| FedAPEN (c8) | 54.83 ± 0.27 / 54.83 (R500, n=3) | 59.74 ± 0.44 / 59.74 (R500, n=3) | 41.84 ± 0.32 / 41.84 (R500, n=3) |
| FedScout, learner every round (500R): scout alone (c8 FedAvg + head FT) | 65.66 ± 0.81 / 65.62 (R500, n=3) | 69.37 ± 1.26 / 68.69 (R500, n=3) | — |
| FedScout, learner every round (500R): scout + log prior | 62.82 ± 0.77 / 62.62 (R500, n=3) | 65.45 ± 1.91 / 65.45 (R500, n=3) | — |
| FedScout, learner every round (500R): fused (scout + α·learner) | 66.23 ± 0.78 / 66.23 (R500, n=3) | 70.15 ± 1.25 / 69.97 (R500, n=3) | — |
| FedScout 2-stage, 490R + 10 local: scout alone (c8 FedAvg + head FT) | 65.73 ± 0.82 / 65.55 (R500, n=3) | 69.32 ± 1.27 / 68.48 (R500, n=3) | — |
| FedScout 2-stage, 490R + 10 local: scout + log prior | 62.96 ± 0.85 / 62.67 (R500, n=3) | 65.47 ± 1.49 / 65.37 (R500, n=3) | — |
| FedScout 2-stage, 490R + 10 local: fused (scout + α·learner) | 66.05 ± 0.91 / 66.05 (R500, n=3) | 69.32 ± 1.27 / 68.21 (R500, n=3) | — |
| FedScout 2-stage, 490R + 50 local (final): scout alone (c8 FedAvg + head FT) | — | 69.31 ± 1.25 / 68.71 (R540, n=3) | — |
| FedScout 2-stage, 490R + 50 local (final): scout + log prior | — | 65.48 ± 1.37 / 65.24 (R540, n=3) | — |
| FedScout 2-stage, 490R + 50 local (final): fused (scout + α·learner) | — | 69.62 ± 0.99 / 69.62 (R540, n=3) | — |

## Paired per-seed differences at the last evaluation (FedScout fused − FedAPEN, same seed and split)

| FedScout config | dataset | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|---|
| FedScout, learner every round (500R) | Cifar100 | +12.17 | +9.99 | +12.04 | +11.40 |
| FedScout, learner every round (500R) | Flowers102 | +11.29 | +7.49 | +11.92 | +10.23 |
| FedScout 2-stage, 490R + 10 local | Cifar100 | +11.77 | +9.78 | +12.11 | +11.22 |
| FedScout 2-stage, 490R + 10 local | Flowers102 | +9.88 | +6.42 | +9.10 | +8.47 |
| FedScout 2-stage, 490R + 50 local (final) | Flowers102 | +10.61 | +7.88 | +11.14 | +9.88 |
