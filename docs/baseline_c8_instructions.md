# 다른 노드 기준선: 공유 모델 구조 (2026-11-18 결정)

**결정: 기본 쌍둥이는 c20(CIFAR ResNet-20)으로 유지한다. 다른 노드는 바꿀 것이 없다.**

- FedAPEN, FedMRL은 원래 계획대로 `--global_arch c20`으로 돌린다. 지금 돌고 있는 c20 run이 본 비교다.
- LG-FedAvg, FedGH, FedProto, FedTGP, FedSA, FedRG, Local은 공유 모델이 없어서 영향 없음.
- c8·c14·c56은 우리 방법의 용량 ablation으로만 쓴다(이 노드에서 돌림).
- Flowers102는 `-lbs 32`여야 한다(클라이언트당 학습 샘플 약 300장). 기준선 로그의 실행 명령에 `-lbs 32`가 있는지 확인한다.

## 경위
- c8(MLPerf Tiny ResNet-8)을 기본으로 검토했으나 Flowers102에서 최고 기준선 +3을 넘지 못했다(150R 57.68, 기준 58.75).
- c14는 기준을 겨우 넘었고(58.92), Flowers에서 c20(62.85)보다 3.9 낮아 c20으로 정했다.
- `--global_arch`는 이제 c8/c14/c20/c56을 모두 받는다(용량 ablation에서 기준선 구조를 맞출 때 사용).
