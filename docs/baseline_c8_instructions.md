# 다른 노드 기준선: 공유 모델 구조 (2026-11-19 변경, 처음 결정은 11-18)

**결정(11-19): 기본 scout은 c8(MLPerf Tiny ResNet-8)이고, 기준선의 공유 모델도 모두 c8로 맞춘다(모든 방법이 같은 용량).**

- 본 표: FedAPEN, pFedMoE, FedMRL(선택: FML)을 `--global_arch c8`로 돌린다. 우선순위는 FedAPEN → pFedMoE(`--pfedmoe_gate_lr 1.0`) → FedMRL.
- 이미 돌린 c20 run(FedAPEN Flowers 68.53·Tiny 48.18·CIFAR 56.95 등)은 부록용이다. 우리 c20 용량 ablation과 나란히 보고한다.
- LG-FedAvg, FedGH, FedProto, FedTGP, FedSA, FedRG, Local은 공유 모델이 없어서 영향 없음.
- Flowers102는 `-lbs 32`여야 한다(클라이언트당 학습 샘플 약 300장). 기준선 로그의 실행 명령에 `-lbs 32`가 있는지 확인한다.

## 경위
- c8(MLPerf Tiny ResNet-8)을 기본으로 검토했으나 Flowers102에서 최고 기준선 +3을 넘지 못했다(150R 57.68, 기준 58.75).
- c14는 기준을 겨우 넘었고(58.92), Flowers에서 c20(62.85)보다 3.9 낮아 c20으로 정했다.
- 11-19: 위 Flowers 차이는 대부분 게이트 조기 동결 탓이었다(c20도 일찍 동결하면 63.5, 늦게 동결하면 71~74). head까지 끝까지 공유하는 구성에서는 c8이 통신(162K vs 358K float/라운드)과 연산(12.5M vs 40.8M MACs)에서 유리하고, 500R 비교 차이는 1.0(CIFAR 60.53 vs 61.53)이라 c8로 바꿨다. 기준선이 c20이면 용량이 어긋나므로 기준선도 c8로 맞춘다.
- `--global_arch`는 이제 c8/c14/c20/c56을 모두 받는다(용량 ablation에서 기준선 구조를 맞출 때 사용).
