# Learner add-on cost vs gain, and cascade accuracy-latency

## Learner on top of the same scout (seed mean, last evaluation)

| dataset (local epochs) | scout alone | fused | learner gain | learner-stage training, TFLOPs/client | Pi 5 learner stage, h/client | Orin learner stage, h/client |
|---|---|---|---|---|---|---|
| Cifar100 (10) | 65.55 | 66.05 | +0.50 | 2.6 | 0.25 | 0.02 |
| Flowers102 (50) | 68.71 | 69.62 | +0.91 | 7.0 | 0.22 | 0.01 |

## Cascade: run the learner only where the scout's top-1 probability is below tau (last evaluation, seed mean)

Latency per sample = scout + run fraction x private (batch 1, mean over the 20 clients' architectures).

| dataset | tau | accuracy | learner run fraction | Pi 5 latency, ms | Orin latency, ms |
|---|---|---|---|---|---|
| Cifar100 | scout only | 65.55 | 0.00 | 2.9 | 5.0 |
| Cifar100 | 0.5 | 66.04 | 0.32 | 16.6 | 14.1 |
| Cifar100 | 0.7 | 66.17 | 0.53 | 25.8 | 20.2 |
| Cifar100 | 0.9 | 66.13 | 0.76 | 35.3 | 26.5 |
| Cifar100 | always (fused) | 66.05 | 1.00 | 45.7 | 33.4 |
| Flowers102 | scout only | 68.71 | 0.00 | 9.3 | 4.9 |
| Flowers102 | 0.5 | 69.81 | 0.32 | 33.7 | 14.4 |
| Flowers102 | 0.7 | 69.81 | 0.52 | 48.3 | 20.1 |
| Flowers102 | 0.9 | 69.63 | 0.72 | 63.8 | 26.2 |
| Flowers102 | always (fused) | 69.62 | 1.00 | 84.6 | 34.3 |
