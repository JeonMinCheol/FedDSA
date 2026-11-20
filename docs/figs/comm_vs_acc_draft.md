# comm 대비 정확도 (초안, 2026-11-19 18:56 KST, `system/comm_vs_acc.py --x comm`)

그림: `comm_vs_acc_draft.png`. FedScout는 본 표 구성(c8 개인 head, 동결 없음)만. 공유 모델 방식(FedScout, FedAPEN, pFedMoE)은 모두 c8(기본이라 범례에 표시하지 않음). FedAPEN 통신량은 실제 공유 모델 state_dict 기준(c8 + 512×C head). 곡선은 seed 평균, 음영은 ±1 seed 표준편차. 연산량은 모델별 실측 FLOPs(`system/flops_table.py`) × 샘플 수.

| dataset | method | best within prototype-500R budget | best within FedAPEN-c8-500R budget | best overall | MB used |
|---|---|---|---|---|---|
| Cifar100 | FedScout (n=3, R500) | 61.02 | 61.73 | 61.73 | 312.3 |
| Cifar100 | pFedMoE (n=1, R500) | 40.40 | 40.92 | 40.92 | 312.3 |
| Cifar100 | FedSA (n=3, R500) | 44.89 | 44.89 | 44.89 | 149.5 |
| Cifar100 | FedTGP (n=3, R500) | 44.51 | 44.51 | 44.51 | 149.5 |
| Cifar100 | FedRG (n=3, R500) | 41.98 | 41.98 | 41.98 | 94.2 |
| Cifar100 | LG-FedAvg (n=3, R500) | 40.48 | 40.64 | 40.64 | 205.2 |
| Cifar100 | FedGH (n=3, R500) | 39.83 | 39.83 | 39.83 | 149.7 |
| Cifar100 | FedProto (n=3, R500) | 38.24 | 38.24 | 38.24 | 149.5 |
| Flowers102 | FedScout (n=3, R500) | 58.98 | 60.05 | 60.05 | 312.3 |
| Flowers102 | FedAPEN (n=3, R500) | 51.71 | 59.74 | 59.74 | 521.6 |
| Flowers102 | FedSA (n=3, R500) | 55.59 | 55.59 | 55.59 | 136.7 |
| Flowers102 | FedTGP (n=3, R500) | 55.60 | 55.60 | 55.60 | 136.7 |
| Flowers102 | FedRG (n=3, R500) | 50.62 | 50.62 | 50.62 | 64.5 |
| Flowers102 | LG-FedAvg (n=3, R500) | 53.69 | 53.80 | 53.80 | 209.3 |
| Flowers102 | FedGH (n=3, R500) | 53.06 | 53.06 | 53.06 | 136.9 |
| Flowers102 | FedProto (n=3, R500) | 49.88 | 49.88 | 49.88 | 136.7 |
| TinyImagenet | FedScout (n=3, R140) | 45.17 | 45.17 | 45.17 | 87.4 |
| TinyImagenet | FedSA (n=3, R500) | 23.92 | 23.92 | 23.92 | 295.7 |
| TinyImagenet | FedTGP (n=3, R500) | 23.20 | 23.20 | 23.20 | 295.7 |
| TinyImagenet | FedRG (n=3, R500) | 24.15 | 24.15 | 24.15 | 181.9 |
| TinyImagenet | LG-FedAvg (n=3, R500) | 23.64 | 23.64 | 23.64 | 410.4 |
| TinyImagenet | FedGH (n=3, R500) | 23.19 | 23.19 | 23.19 | 296.1 |
| TinyImagenet | FedProto (n=3, R500) | 18.24 | 18.24 | 18.24 | 295.7 |
budgets (MB per client): Cifar100: prototype 149.5, FedAPEN-c8 517.5, Flowers102: prototype 136.7, FedAPEN-c8 521.6, TinyImagenet: prototype 295.7, FedAPEN-c8 722.7
