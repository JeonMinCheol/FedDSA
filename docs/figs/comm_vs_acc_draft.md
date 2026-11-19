# 통신량 대비 정확도 (초안, 2026-11-19 15:00 KST, `system/comm_vs_acc.py`)

그림: `comm_vs_acc_draft.png`. 기준선은 500R 결과 하나뿐이라 점, FedScout는 seed 평균 곡선. 예산 안 최고값은 seed 평균 곡선에서 누적 MB가 예산 이하인 라운드들의 최고값.

| dataset | method | best within prototype-500R budget | best within FedAPEN-c20-500R budget | best overall | MB used |
|---|---|---|---|---|---|
| Cifar100 | FedScout c20, personal head (n=3, R500) | 56.27 | 61.54 | 61.54 | 365 |
| Cifar100 | FedScout c20, share-all (n=3, R500) | 46.75 | 64.46 | 64.46 | 685 |
| Cifar100 | FedScout c8, share-all (seed 0) (n=1, R330) | 45.09 | 54.93 | 54.93 | 429 |
| Cifar100 | FedProto (500R) | 38.24 | 38.24 | 38.24 | 150 |
| Cifar100 | FedGH (500R) | 39.83 | 39.83 | 39.83 | 150 |
| Cifar100 | LG-FedAvg (500R) | - | 40.64 | 40.64 | 205 |
| Cifar100 | FedRG (500R) | 41.98 | 41.98 | 41.98 | 94 |
| Cifar100 | FedTGP (500R) | 44.51 | 44.51 | 44.51 | 150 |
| Cifar100 | FedSA (500R) | 45.02 | 45.02 | 45.02 | 150 |
| Cifar100 | FedAPEN (c20) (500R) | - | 56.95 | 56.95 | 1113 |
| Flowers102 | FedScout c20, personal head (n=3, R500) | 59.43 | 61.72 | 61.72 | 235 |
| Flowers102 | FedScout c20, share-all (n=3, R500) | 56.09 | 68.64 | 68.64 | 525 |
| Flowers102 | FedScout c8, share-all (seed 0) (n=1, R500) | 57.08 | 65.99 | 65.99 | 643 |
| Flowers102 | FedProto (500R) | 49.88 | 49.88 | 49.88 | 137 |
| Flowers102 | FedRG (500R) | 50.62 | 50.62 | 50.62 | 65 |
| Flowers102 | FedGH (500R) | 53.06 | 53.06 | 53.06 | 137 |
| Flowers102 | LG-FedAvg (500R) | - | 53.80 | 53.80 | 209 |
| Flowers102 | FedTGP (500R) | 55.60 | 55.60 | 55.60 | 137 |
| Flowers102 | FedSA (500R) | 55.75 | 55.75 | 55.75 | 137 |
| Flowers102 | FedAPEN (c20) (500R) | - | 68.53 | 68.53 | 1114 |
| TinyImagenet | FedScout c20, personal head (n=3, R500) | 50.72 | 52.09 | 52.09 | 472 |
| TinyImagenet | FedProto (500R) | 18.24 | 18.24 | 18.24 | 296 |
| TinyImagenet | FedGH (500R) | 23.19 | 23.19 | 23.19 | 296 |
| TinyImagenet | FedTGP (500R) | 23.20 | 23.20 | 23.20 | 296 |
| TinyImagenet | LG-FedAvg (500R) | - | 23.64 | 23.64 | 410 |
| TinyImagenet | FedSA (500R) | 23.88 | 23.88 | 23.88 | 296 |
| TinyImagenet | FedRG (500R) | 24.15 | 24.15 | 24.15 | 182 |
| TinyImagenet | FedAPEN (c20) (500R) | - | 48.18 | 48.18 | 1139 |
budgets (MB per client): Cifar100: prototype 150, FedAPEN-c20 1113, Flowers102: prototype 137, FedAPEN-c20 1114, TinyImagenet: prototype 296, FedAPEN-c20 1139
