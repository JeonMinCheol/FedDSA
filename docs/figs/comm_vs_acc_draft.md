# 통신량 대비 정확도 (초안, 2026-11-19 16:07 KST, `system/comm_vs_acc.py`)

그림: `comm_vs_acc_draft.png`. FedScout와 곡선이 있는 기준선(TSV, `docs/figs/data/`)은 seed 평균 곡선, 나머지 기준선은 500R 점. 예산 안 최고값은 누적 MB가 예산 이하인 라운드들의 seed 평균 곡선 최고값.

| dataset | method | best within prototype-500R budget | best within FedAPEN-c20-500R budget | best overall | MB used |
|---|---|---|---|---|---|
| Cifar100 | FedScout c20, personal head (n=3, R500) | 56.27 | 61.54 | 61.54 | 365 |
| Cifar100 | FedScout c20, share-all (n=3, R500) | 46.75 | 64.46 | 64.46 | 685 |
| Cifar100 | FedScout c8, share-all (seed 0) (n=1, R500) | 45.09 | 64.30 | 64.30 | 639 |
| Cifar100 | pFedMoE (c8) (n=1, R500) | 40.40 | 40.92 | 40.92 | 312 |
| Cifar100 | FedSA (n=3, R500) | 44.89 | 44.89 | 44.89 | 150 |
| Cifar100 | FedTGP (n=3, R500) | 44.51 | 44.51 | 44.51 | 150 |
| Cifar100 | FedAPEN (c20) (n=3, R500) | 42.42 | 56.95 | 56.95 | 1113 |
| Cifar100 | FedProto (500R) | 38.24 | 38.24 | 38.24 | 150 |
| Cifar100 | FedGH (500R) | 39.83 | 39.83 | 39.83 | 150 |
| Cifar100 | LG-FedAvg (500R) | - | 40.64 | 40.64 | 205 |
| Cifar100 | FedRG (500R) | 41.98 | 41.98 | 41.98 | 94 |
| Flowers102 | FedScout c20, personal head (n=3, R500) | 59.43 | 61.72 | 61.72 | 235 |
| Flowers102 | FedScout c20, share-all (n=3, R500) | 56.09 | 68.64 | 68.64 | 525 |
| Flowers102 | FedScout c8, share-all (seed 0) (n=1, R500) | 57.08 | 65.99 | 65.99 | 643 |
| Flowers102 | FedSA (n=3, R500) | 55.59 | 55.59 | 55.59 | 137 |
| Flowers102 | FedAPEN (c20) (n=3, R500) | 48.39 | 68.53 | 68.53 | 1114 |
| Flowers102 | FedAPEN (c8) (n=3, R500) | 53.79 | 59.74 | 59.74 | 336 |
| Flowers102 | FedProto (500R) | 49.88 | 49.88 | 49.88 | 137 |
| Flowers102 | FedRG (500R) | 50.62 | 50.62 | 50.62 | 65 |
| Flowers102 | FedGH (500R) | 53.06 | 53.06 | 53.06 | 137 |
| Flowers102 | LG-FedAvg (500R) | - | 53.80 | 53.80 | 209 |
| Flowers102 | FedTGP (500R) | 55.60 | 55.60 | 55.60 | 137 |
| TinyImagenet | FedScout c20, personal head (n=3, R500) | 50.72 | 52.09 | 52.09 | 472 |
| TinyImagenet | FedSA (n=3, R500) | 23.92 | 23.92 | 23.92 | 296 |
| TinyImagenet | FedRG (n=3, R500) | 24.15 | 24.15 | 24.15 | 182 |
| TinyImagenet | FedAPEN (c20) (n=3, R500) | 36.66 | 48.31 | 48.31 | 1139 |
| TinyImagenet | FedProto (500R) | 18.24 | 18.24 | 18.24 | 296 |
| TinyImagenet | FedGH (500R) | 23.19 | 23.19 | 23.19 | 296 |
| TinyImagenet | FedTGP (500R) | 23.20 | 23.20 | 23.20 | 296 |
| TinyImagenet | LG-FedAvg (500R) | - | 23.64 | 23.64 | 410 |
budgets (MB per client): Cifar100: prototype 150, FedAPEN-c20 1113, Flowers102: prototype 137, FedAPEN-c20 1114, TinyImagenet: prototype 296, FedAPEN-c20 1139
