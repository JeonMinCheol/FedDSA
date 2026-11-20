# compute 대비 정확도 (초안, 2026-11-19 18:56 KST, `system/comm_vs_acc.py --x compute`)

그림: `compute_vs_acc_draft.png`. FedScout는 본 표 구성(c8 개인 head, 동결 없음)만. 공유 모델 방식(FedScout, FedAPEN, pFedMoE)은 모두 c8(기본이라 범례에 표시하지 않음). FedAPEN 통신량은 실제 공유 모델 state_dict 기준(c8 + 512×C head). 곡선은 seed 평균, 음영은 ±1 seed 표준편차. 연산량은 모델별 실측 FLOPs(`system/flops_table.py`) × 샘플 수.

| dataset | method | best within prototype-500R budget | best within FedAPEN-c8-500R budget | best overall | TFLOPs used |
|---|---|---|---|---|---|
| Cifar100 | FedScout (n=3, R500) | 61.73 | 61.73 | 61.73 | 304.8 |
| Cifar100 | pFedMoE (n=1, R500) | 40.92 | 40.92 | 40.92 | 544.5 |
| Cifar100 | FedSA (n=3, R500) | 44.89 | 44.89 | 44.89 | 460.8 |
| Cifar100 | FedTGP (n=3, R500) | 44.51 | 44.46 | 44.51 | 615.9 |
| Cifar100 | FedRG (n=3, R500) | 41.98 | 41.98 | 41.98 | 615.9 |
| Cifar100 | LG-FedAvg (n=3, R500) | 40.64 | 40.64 | 40.64 | 460.8 |
| Cifar100 | FedGH (n=3, R500) | 39.83 | 39.80 | 39.83 | 615.9 |
| Cifar100 | FedProto (n=3, R500) | 38.24 | 38.24 | 38.24 | 460.8 |
| Flowers102 | FedScout (n=3, R500) | 60.05 | 60.05 | 60.05 | 115.4 |
| Flowers102 | FedAPEN (n=3, R500) | 59.74 | 59.74 | 59.74 | 290.3 |
| Flowers102 | FedSA (n=3, R500) | 55.59 | 55.59 | 55.59 | 244.8 |
| Flowers102 | FedTGP (n=3, R500) | 55.60 | 55.60 | 55.60 | 327.2 |
| Flowers102 | FedRG (n=3, R500) | 50.62 | 50.40 | 50.62 | 327.2 |
| Flowers102 | LG-FedAvg (n=3, R500) | 53.80 | 53.80 | 53.80 | 244.8 |
| Flowers102 | FedGH (n=3, R500) | 53.06 | 52.72 | 53.06 | 327.2 |
| Flowers102 | FedProto (n=3, R500) | 49.88 | 49.88 | 49.88 | 244.8 |
| TinyImagenet | FedScout (n=3, R140) | 45.17 | 45.17 | 45.17 | 923.1 |
| TinyImagenet | FedSA (n=3, R500) | 23.92 | 23.92 | 23.92 | 3082.3 |
| TinyImagenet | FedTGP (n=3, R500) | 23.20 | 22.67 | 23.20 | 4119.9 |
| TinyImagenet | FedRG (n=3, R500) | 24.15 | 24.08 | 24.15 | 4119.9 |
| TinyImagenet | LG-FedAvg (n=3, R500) | 23.64 | 23.64 | 23.64 | 3082.3 |
| TinyImagenet | FedGH (n=3, R500) | 23.19 | 23.19 | 23.19 | 4119.9 |
| TinyImagenet | FedProto (n=3, R500) | 18.24 | 18.24 | 18.24 | 3082.3 |
budgets (TFLOPs per client): Cifar100: prototype 615.9, FedAPEN-c8 544.5, Flowers102: prototype 327.2, FedAPEN-c8 290.3, TinyImagenet: prototype 4119.9, FedAPEN-c8 3639.3
