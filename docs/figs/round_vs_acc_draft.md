# round 대비 정확도 (초안, 2026-11-19 18:56 KST, `system/comm_vs_acc.py --x round`)

그림: `round_vs_acc_draft.png`. FedScout는 본 표 구성(c8 개인 head, 동결 없음)만. 공유 모델 방식(FedScout, FedAPEN, pFedMoE)은 모두 c8(기본이라 범례에 표시하지 않음). FedAPEN 통신량은 실제 공유 모델 state_dict 기준(c8 + 512×C head). 곡선은 seed 평균, 음영은 ±1 seed 표준편차. 연산량은 모델별 실측 FLOPs(`system/flops_table.py`) × 샘플 수.

| dataset | method | best within prototype-500R budget | best within FedAPEN-c8-500R budget | best overall | rounds used |
|---|---|---|---|---|---|
| Cifar100 | FedScout (n=3, R500) | 61.73 | 61.73 | 61.73 | 500.0 |
| Cifar100 | pFedMoE (n=1, R500) | 40.92 | 40.92 | 40.92 | 500.0 |
| Cifar100 | FedSA (n=3, R500) | 44.89 | 44.89 | 44.89 | 500.0 |
| Cifar100 | FedTGP (n=3, R500) | 44.51 | 44.51 | 44.51 | 500.0 |
| Cifar100 | FedRG (n=3, R500) | 41.98 | 41.98 | 41.98 | 500.0 |
| Cifar100 | LG-FedAvg (n=3, R500) | 40.64 | 40.64 | 40.64 | 500.0 |
| Cifar100 | FedGH (n=3, R500) | 39.83 | 39.83 | 39.83 | 500.0 |
| Cifar100 | FedProto (n=3, R500) | 38.24 | 38.24 | 38.24 | 500.0 |
| Flowers102 | FedScout (n=3, R500) | 60.05 | 60.05 | 60.05 | 500.0 |
| Flowers102 | FedAPEN (n=3, R500) | 59.74 | 59.74 | 59.74 | 500.0 |
| Flowers102 | FedSA (n=3, R500) | 55.59 | 55.59 | 55.59 | 500.0 |
| Flowers102 | FedTGP (n=3, R500) | 55.60 | 55.60 | 55.60 | 500.0 |
| Flowers102 | FedRG (n=3, R500) | 50.62 | 50.62 | 50.62 | 500.0 |
| Flowers102 | LG-FedAvg (n=3, R500) | 53.80 | 53.80 | 53.80 | 500.0 |
| Flowers102 | FedGH (n=3, R500) | 53.06 | 53.06 | 53.06 | 500.0 |
| Flowers102 | FedProto (n=3, R500) | 49.88 | 49.88 | 49.88 | 500.0 |
| TinyImagenet | FedScout (n=3, R140) | 45.17 | 45.17 | 45.17 | 140.0 |
| TinyImagenet | FedSA (n=3, R500) | 23.92 | 23.92 | 23.92 | 500.0 |
| TinyImagenet | FedTGP (n=3, R500) | 23.20 | 23.20 | 23.20 | 500.0 |
| TinyImagenet | FedRG (n=3, R500) | 24.15 | 24.15 | 24.15 | 500.0 |
| TinyImagenet | LG-FedAvg (n=3, R500) | 23.64 | 23.64 | 23.64 | 500.0 |
| TinyImagenet | FedGH (n=3, R500) | 23.19 | 23.19 | 23.19 | 500.0 |
| TinyImagenet | FedProto (n=3, R500) | 18.24 | 18.24 | 18.24 | 500.0 |
budgets (rounds per client): Cifar100: prototype 500.0, FedAPEN-c8 500.0, Flowers102: prototype 500.0, FedAPEN-c8 500.0, TinyImagenet: prototype 500.0, FedAPEN-c8 500.0
