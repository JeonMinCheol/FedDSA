# 논문 스토리와 실험 구성 (2026-11-19 밤 기준)

목표: IEEE 저널(IoT-J급). 실험 규칙·프로토콜은 `docs/experiment_plan.md`, 진행 기록은 `system/logs/night_actions_20261117.md`, 기준선 측정은 `docs/baselines_progress.md`.

## 1. 한 줄

**이기종 FL에서 지식이 실제로 어디로 흐르는지 밝히고, 그에 맞춘 싼 설계(FedScout)를 제안한다.**
노벨티는 새 구조가 아니라 발견과 효율에 건다. 정확도는 같은 용량(c8) 기준선을 이기면 충분하고, 표적은 통신·연산·분석이다.

## 2. 스토리 (논문 흐름)

1. **문제 설정.** 여러 제조사 IoT 기기가 각자 다른 모델을 쓴다(이기종). 데이터 증강을 못 하고, label skew가 극단적이며, 통신·연산이 비싸다.
2. **관찰 1: 지식은 같은 구조끼리만 흐른다.** 구조가 다른 모델 사이의 채널(prototype, anchor, logit, 생성기)을 쓰는 HtFL 기준선은 Local과 차이가 작다. 같은 구조의 가중치를 평균할 때만 지식이 옮겨간다.
3. **관찰 2: 이기종 private 모델은 지역 클래스 분포만 더한다.** 공유 모델 + private 앙상블(FedAPEN)의 이득은 "공유 모델 + log 지역 클래스 비율"로 재현되거나 그보다 못하다. 우리 방법에서도 private learner의 기여는 0에 가깝다.
4. **관찰 3: 작은 공유 모델도 결국 암기한다.** 후반에 정확도가 내려간다. 기준선에서도 보인다.
5. **설계 FedScout.** 관찰에 맞춰, 지식 전달은 모든 기기가 같은 구조로 갖는 작은 scout(c8)에 맡긴다. 이기종 private 모델은 scout 위의 잔차로만 두고, scout이 이미 맞히는 샘플은 학습과 추론에서 건너뛰어 private 비용을 줄인다.
6. **결과.** 같은 용량의 기준선보다 정확도가 높거나 같고, 같은 정확도를 훨씬 적은 통신과 연산으로 낸다. 기기 실측으로 확인한다.

## 3. 기여 (논문 서론용)

| # | 기여 | 근거 실험 |
|---|---|---|
| 1 | 이기종 FL의 지식 흐름 분석: 같은 구조 평균만 전달(관찰 1), private은 분포만 더함(관찰 2), 공유 모델의 암기(관찰 3) | 분석 표, 기준선 곡선, ablation(평균 없음) |
| 2 | FedScout: 작은 동종 scout + 개인 head + 잔차 learner + screening/cascade | 본 표, ablation |
| 3 | 효율: 같은 정확도를 공유 모델 방식보다 적은 통신(FedAPEN의 60%, 58점 도달 기준 약 1/9)과 연산(약 40%)으로, 추론 때 private 실행 10~25% | 통신·연산 그림, 효율 표, 기기 실측 |
| 4 | 일반성: 클라이언트 수·참여율, 모델 이질성 정도, 저사양 비중에서도 유지 | 일반화 표 |

## 4. 최종 구성 (FedScout)

c8 scout(MLPerf Tiny ResNet-8 크기), backbone만 FedAvg + 개인 head(매 라운드 재정렬), pool projector(HtFLlib과 같음), P2 잔차 learner(결합 = scout + α·learner, α는 held-out으로 선택), screening τ=0.5, cascade 추론, 동결 없음(멈춤은 통신량 대비 곡선으로 보고, patience 게이트는 부록), scout lr 0.03·private lr 0.01(우리 하이퍼파라미터; 기준선 lr은 원 논문 0.01 그대로).

## 5. 넣을 실험과 진행 상태

### 5.1 본문

| 실험 | 내용 | 상태 |
|---|---|---|
| **본 표** | 3 데이터셋 × 3 seed, 500R. 열: 최종 정확도(seed 평균 곡선 최고), prototype 방식 500R 예산에서의 정확도, 총 통신량·연산량 | FedScout pool: CIFAR·Flowers 진행 중(오늘 밤 완료), Tiny seed 0 진행, 1·2 대기열 맨 앞 |
| 기준선 (같은 프로토콜) | LG-FedAvg, FedGH, FedProto, FedTGP, FedSA(λ=1), FedRG: 3 데이터셋 곡선 모두 확보(문서 값 = 프로토콜 값 확인) | 완료 |
| 기준선 (공유 모델 방식, c8, 원 논문 방식) | FedAPEN, pFedMoE, FedMRL | FedAPEN Flowers 완료(59.74), Tiny 01시 완료 예정, CIFAR 대기. pFedMoE CIFAR s0 완료(40.9), 나머지 진행. FedMRL 대기 (모두 ariel) |
| Local | 관찰 1의 기준점 | 다른 노드에서 실행했다고 기록됨. **결과 확인 필요** |
| **그림 3종** | 정확도 대 라운드 / 통신량 / 학습 연산량 (seed 평균 ± 표준편차) | 초안 완료(`docs/figs/*_vs_acc_draft.png`). 본 표 run이 끝나면 다시 그림 |
| **효율 표** | 라운드당 통신, 목표 정확도까지의 통신, 학습 FLOPs(모델별 실측), 추론 연산(cascade로 private 실행 비율) | 계산 스크립트 완료(`system/comm_vs_acc.py`, `system/flops_table.py`). 표 정리만 남음 |
| **기기 실측** | Jetson Nano, Raspberry Pi에서 라운드당 학습 시간·에너지·메모리, 추론 지연 | **사용자가 직접 측정** (`system/device_bench.py`) |
| **분석 표 (관찰 2)** | FedAPEN: 앙상블 / private / 공유 / 공유 + 비율. FedScout: 결합 / scout 단독 / learner | 로그로 작성 가능(run 불필요). Tiny FedAPEN: 공유 + 비율 45.7 > 앙상블 39.1, FedScout 결합 = scout |
| **ablation (CIFAR × 3 seed)** | screening 끔 / 재정렬 끔 / I1 learner / head까지 평균(shareall, R490에 개인화) / 평균 없음 / scout lr×1 | 대기열(우선순위 3번째) |
| 일반화: 클라이언트 수·참여율 | 50명 join 0.5, 100명 join 0.5·0.3·0.1 × 3 seed | 대기열. 기준선은 ariel이 같은 설정으로 측정 중 |
| 일반화: 모델 이질성 | HtFE 2·3·4·5·9, lowend × 3 seed | 대기열. 기준선은 예전 원고 표 1 재사용 예정 → **프로토콜 일치 확인 필요** |

### 5.2 부록

| 실험 | 상태 |
|---|---|
| FedAPEN + 개인 head(lr 0.01): "우리 발견을 기존 방법에 적용" | 진행 중(CIFAR 3개 실행, Flowers 3개 대기열) |
| 용량 표: c8 vs c20 (FedScout c20 seed 0, FedAPEN c20 3 데이터셋 완료) | FedScout c20 대기열 |
| shareall 변형(정확도 더 높음, 통신 2배) | ablation run으로 대체 |
| pool vs linear projector | linear CIFAR·Flowers 3 seed 완료, pool 진행 중 |
| screening 임계값 τ 0.3 / 0.8 | 대기열 |
| patience 게이트로 자동 멈춤했을 때(통신 절감과 정확도) | 본 표 로그의 held-out으로 사후 계산(run 불필요) |
| 로컬 에폭 E=2(500R) / 5(200R) / 10(100R) | 대기열 맨 끝, 필요 없으면 생략 |

### 5.3 하지 않는 것
FedDSA(이중 앵커), 동결 후 회복, 합성 이미지 증류, head 고정(FedBABU 방식), 동결 때 head 한 번 평균, 클래스 분포 보정 screening(정확도 손실로 폐기), lr×3을 준 기준선(본문에서 제외). label skew 강도(Dir α 0.3/1.0)는 분할 데이터가 없어 보류(결정 필요).

## 6. 리뷰 대비 (위험과 답)

| 예상 지적 | 답 |
|---|---|
| "작은 c8에 FedPer(개인 head)를 붙인 것 아니냐, private은 왜 두나" | 관찰 2를 숨기지 않고 분석의 중심에 둔다. 설정 전제(기기는 자기 모델을 유지)와, 그 전제에서 private 비용을 최소화하는 것이 FedScout의 몫. 분석 표에 scout 단독 줄을 명시 |
| "우리만 lr을 튜닝했다" | 기준선 lr은 원 논문 값(세 논문 모두 0.01). ablation의 scout lr×1, 부록의 FedAPEN + 개인 head로 답함 |
| "projector 덕분 아니냐" | pool로 고정(기준선과 동일) |
| "통신 이점이 없다" (500R 총량은 개인 head 기준선과 같음) | 근거를 "목표 정확도까지의 통신"으로 명확히. FedAPEN 원본 대비로는 총량도 60% |
| "후반에 떨어진다" (CIFAR R350 → R500 약 −1.2) | 관찰 3과 연결, held-out 최고점 복원·patience 사후 분석으로 답 |
| "실제 기기에서도 빠른가" | 기기 실측 (필수) |
