# compute 대비 정확도 (초안, 2026-11-19 18:15 KST, `system/comm_vs_acc.py --x compute`)

그림: `compute_vs_acc_draft.png`. 공유 모델 방식은 c8만. 곡선은 seed 평균, 음영은 ±1 seed 표준편차. 연산량은 모델별 실측 FLOPs(`system/flops_table.py`) × 샘플 수, 집계 방식은 스크립트 docstring. FedScout 본 표 run(main_c8)은 진행 중이면 seed마다 도달한 라운드까지만 평균.

| dataset | method | best within prototype-500R budget | best within FedAPEN-c8-500R budget | best overall | TFLOPs used |
|---|---|---|---|---|---|
| Cifar100 | FedScout c8 (personal head) (n=3, R470) | 61.73 | 61.73 | 61.73 | 294.4 |
| Cifar100 | FedScout c8, share-all (appendix) (n=1, R500) | 57.96 | 57.96 | 57.96 | 517.1 |
| Cifar100 | pFedMoE (c8) (n=1, R500) | 40.92 | 40.92 | 40.92 | 544.5 |
| Cifar100 | FedSA (n=3, R500) | 44.89 | 44.89 | 44.89 | 460.8 |
| Cifar100 | FedTGP (n=3, R500) | 44.51 | 44.46 | 44.51 | 615.9 |
| Cifar100 | LG-FedAvg (n=3, R500) | 40.64 | 40.64 | 40.64 | 460.8 |
| Cifar100 | FedGH (n=3, R500) | 39.83 | 39.80 | 39.83 | 615.9 |
| Cifar100 | FedProto (n=3, R500) | 38.24 | 38.24 | 38.24 | 460.8 |
| Cifar100 | FedRG (500R) | 41.98 | - | 41.98 | 615.9 |
| Flowers102 | FedScout c8 (personal head) (n=3, R500) | 60.05 | 60.05 | 60.05 | 115.4 |
| Flowers102 | FedScout c8, share-all (appendix) (n=1, R500) | 64.72 | 64.72 | 64.72 | 268.1 |
| Flowers102 | FedScout c8, personal head, patience 40 (n=1, R500) | 58.35 | 58.35 | 58.35 | 87.8 |
| Flowers102 | FedAPEN (c8) (n=3, R500) | 59.74 | 59.74 | 59.74 | 290.3 |
| Flowers102 | FedSA (n=3, R500) | 55.59 | 55.59 | 55.59 | 244.8 |
| Flowers102 | FedTGP (n=3, R500) | 55.60 | 55.60 | 55.60 | 327.2 |
| Flowers102 | FedRG (n=3, R500) | 50.62 | 50.40 | 50.62 | 327.2 |
| Flowers102 | LG-FedAvg (n=3, R500) | 53.80 | 53.80 | 53.80 | 244.8 |
| Flowers102 | FedGH (n=3, R500) | 53.06 | 52.72 | 53.06 | 327.2 |
| Flowers102 | FedProto (n=3, R500) | 49.88 | 49.88 | 49.88 | 244.8 |
| TinyImagenet | FedScout c8 (personal head) (n=2, R80) | 41.66 | 41.66 | 41.66 | 556.4 |
| TinyImagenet | FedSA (n=3, R500) | 23.92 | 23.92 | 23.92 | 3082.3 |
| TinyImagenet | FedTGP (n=3, R500) | 23.20 | 22.67 | 23.20 | 4119.9 |
| TinyImagenet | FedRG (n=3, R500) | 24.15 | 24.08 | 24.15 | 4119.9 |
| TinyImagenet | LG-FedAvg (n=3, R500) | 23.64 | 23.64 | 23.64 | 3082.3 |
| TinyImagenet | FedGH (n=3, R500) | 23.19 | 23.19 | 23.19 | 4119.9 |
| TinyImagenet | FedProto (n=3, R500) | 18.24 | 18.24 | 18.24 | 3082.3 |
budgets (TFLOPs per client): Cifar100: prototype 615.9, FedAPEN-c8 544.2, Flowers102: prototype 327.2, FedAPEN-c8 290.3, TinyImagenet: prototype 4119.9, FedAPEN-c8 3638.3
