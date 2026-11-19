# 실험 계획: 쌍둥이 잔차 방법 논문 (2026-11-18 작성)

목표: IoT-J급 IEEE 저널. 실험 규칙과 기존 결과는 `docs/twin_handoff.md`, 매일의 판정은 `system/logs/night_actions_20261117.md`.

## 0. 논문 주장과 그것을 받치는 실험

| 주장 | 내용 | 받치는 실험 |
|---|---|---|
| C1 진단 | 증강이 없는 HtFL에서 구조 간 채널(프로토타입, 앵커, logit, 생성기)은 Local과 같다. 같은 구조 가중치 평균만 지식을 옮긴다 | 3절 본 표의 Local·HtFL 기준선, 6절 진단 |
| C2 방법 | 공유 쌍둥이(안정) + 이기종 private 잔차(가소) + 암기 게이트 + 재정렬 + 전역 held-out α | 3절 본 표, 4절 ablation |
| C3 효율 | 동결 후 통신 0, 쌍둥이 게이트로 private 학습·추론 연산 절감, 동결 후 특징 캐시 | 7절 통신·기기 측정 |
| (부록) 배포 | 원래 모델만 배포해야 할 때는 I1 변형(`--scout_residual i1`)으로 약 43(Local·HtFL 수준, F1과 일치). 본 주장에서 제외(11-19 사용자 결정: 결합 logit이 본 결과). 동결 후 회복은 폐기(결합 이득 없음, screening과 충돌, I1이 더 높음) | 4절 P2 vs I1 ablation |

FedDSA(이중 앵커)는 쓰지 않는다(11-18 사용자 결정). 논문에서도 다루지 않는다.

## 1. 고정 프로토콜 (모든 run 공통, 예전 원고 reference/main.pdf와 같음)

- HtFLlib, HtFE8(`-m HtFE-img-8`), 20 clients, Dir α=0.1, join ratio 1.0, 500R, local epoch 1, SGD lr 0.01, `-fd 512`
- 배치: CIFAR-100·Tiny 128, Flowers102 32. 증강 없음(`--fedsa_augment false`)
- seed 0, 1, 2. 10라운드마다 평가. 보고 값 = seed 평균 곡선의 최고값(그 라운드의 표준편차 함께)
- 판단 규칙: 같은 seed 재실행 잡음이 0.3~0.8이므로 단일 seed에서 1.0 미만 차이로 결정하지 않는다
- 실행 규칙(GPU당 2개, Tiny는 1개, `-sfn temp_<이름>`)은 handoff 6절

## 2. 0단계: 최종 구성 확정 (11-19 오전까지)

진행 중인 판단과 사전 규칙. 모두 CIFAR-100 seed 0.

| 항목 | run | 채택 기준 | 상태 |
|---|---|---|---|
| 쌍둥이 lr ×3 vs ×4 | p500_s0, p500lr4_s0 | ×4가 최고값 +1.0 이상 | 중간 +0.9 → ×3 유지 쪽 |
| 쌍둥이 게이트 private 학습 τ | pg05_s0, pg08_s0 | 절감 40% 이상, 결합 손실 0.5 이하 | τ=0.5 통과(62.19, 절감 49.6%) |
| 동결 후 head 고정 | p500fh_s0 vs p500_s0 | 동결 후 하락폭 절반 이하 | 실행 중 |
| 동결 후 회복 | rcv_s0, rcv_ctrl_s0, rcvf_s0, rcvT4_s0 | 원래 모델 단독 40 이상, 결합 −0.3 이내 | 기준은 통과(40.76)했으나 폐기: 결합 이득 없음, T=4도 +0.41, I1 변형(43)이 대신함 |

확정 후 기본 구성(예상): c20, lr×3, 게이트 20, 재정렬 1, P2, 전역 α, 특징 캐시, 쌍둥이 게이트 τ=0.5(학습+추론, 동결 후에도 유지), head 고정(판정 결과). 회복은 폐기(원래 모델만 배포하는 경우는 I1 변형으로 보고).
확정 전에 본 표를 시작하지 않는다. 바뀌는 항목이 있으면 그 항목을 켠 구성으로 seed 0을 다시 돌린다.

## 3. 본 표 (데이터셋 3개 × seed 3개, 500R)

### 우리 방법 (이 노드): 9 run
CIFAR-100, Flowers102, Tiny-ImageNet × seed 0~2. 한 run에서 세 값을 모두 기록한다.
- 결합값(쌍둥이 + α·private): 쌍둥이와 함께 배포할 때
- 쌍둥이 단독: FedRep식 "쌍둥이만" 기준선 역할

### 기준선 (다른 노드, 같은 프로토콜)
| 묶음 | 방법 | 비교하는 배포 상황 | 상태 |
|---|---|---|---|
| HtFL (원래 모델만) | LG-FedAvg, FedGH, FedProto, FedTGP, FedSA(논문 공식 λ=1·MCL 1·CC 1 = 코드 기본값, 본 표 값은 이 설정으로 잰 것), FedRG | 원래 모델만 | 3 데이터셋 측정 완료 |
| 하한 | **Local** | 원래 모델만 | **아직 없음, 최우선** (C1 핵심) |
| 공유 모델 계열 | **FedAPEN 원본**(원 논문 방식, 공유 모델 구조만 c20, `-algo FedAPEN --global_arch c20`) | 함께 배포 | Flowers 68.53·Tiny 48.18 완료, **CIFAR 3 seed 최우선** |
| 공유 모델 계열 | **pFedMoE 원본**(원 논문: 두 extractor 같은 lr, gate lr은 논문 그리드 중 최선 1.0, `-algo pFedMoE --global_arch c20 --pfedmoe_gate_lr 1.0`) | 함께 배포 | 3 데이터셋 × 3 seed 필요 |
| 공유 모델 계열 | FedMRL(`--global_arch c20`) | 함께 배포 | 실행 중 |

**본 표의 기준선은 원 논문 방식 그대로 쓴다(11-19 사용자 결정).** 이기종 FL에 옮기면서 꼭 필요한 조정(공유 모델 구조를 우리 scout과 같은 c20으로)만 하고 논문에 밝힌다. 우리 원칙을 붙인 변형은 본 표가 아니라 부록/분석 절에 둔다("우리 발견이 다른 방법에도 통한다", "기준선을 튜닝해도 우리가 이긴다"):
- FedAPEN + 개인 head + 공유 lr×3(`--apen_personal_head true --apen_shared_lr_scale 3`, 다른 노드 실행 중)
- pFedMoE + 공유 extractor lr×3(`--pfedmoe_shared_lr_scale 3 --pfedmoe_gate_lr 1.0`, 150R 46.93)
| 선택 | FML(`--global_arch c20`) | 함께 배포 | 리뷰어가 물을 가능성 있음, CIFAR만이라도 |

표 구성: 열 = 데이터셋, 행 = 방법, 우리 방법은 "결합(본 결과) / 쌍둥이 단독" 두 줄. 원래 모델만 배포하는 경우(I1 변형)는 부록.

## 4. Ablation (CIFAR-100, 500R)

seed 3개로 할 핵심 항목(본 문에 들어감). 하나씩만 끈다.

| 항목 | 끄는 방법 | 확인하는 것 |
|---|---|---|
| 게이트 | `--fedsa_cm_gate 0` | 통신 절감 대비 정확도, 후반 하락 |
| 재정렬 | `--fedsa_cm_realign 0` | F5(head 표류) |
| 잔차 방식 | I1(`--fedsa_res_standalone true`) | P2 vs I1: 결합 vs 원래 모델 단독 |
| 쌍둥이 게이트 | `--fedsa_priv_gate 0` | 연산 절감이 정확도 손실 없이 나오는지 |
| 쌍둥이 head 평균 | `--fedsa_cm_base_only false` | F3(head 평균 붕괴) |
| 평균 없음 | 쌍둥이를 평균하지 않음 | F2(같은 구조 평균만 지식을 옮김) |

seed 1개로 할 보조 항목(부록): lr ×2/×4, weight decay, BN 로컬, 쌍둥이 용량 c8/c14/c20/c56, τ ∈ {0.3, 0.5, 0.8}, 고정 라운드 동결(`--fedsa_cm_freeze_at`) vs 게이트.
이미 150R·500R 단일 seed 결과가 있는 항목은 그대로 쓰고, 500R이 없는 것만 돌린다.

## 5. 일반화 (CIFAR-100, 500R)

예전 원고(reference/main.pdf) 표 1·3과 같은 설정이라 그 HtFL 기준선 수치를 재사용한다. 단 그 원고의 FedSA 행은 λ=10으로 잰 값이라 논문 공식값 λ=1(`-lam 1`)로 다시 측정해야 한다.

| 실험 | 설정 | 우리 run | 기준선 |
|---|---|---|---|
| 모델 이질성 | HtFE2/3/4/5/9 (`-m HtFE-img-X`) | 5 × 3 seed | 예전 원고 표 1 재사용 + FedSA·Local·FedAPEN-ph 새로 |
| 저사양 비중 (IoT) | `HtFE-img-lowend` | 3 seed | 새로: Local, FedTGP, FedSA, FedAPEN-ph |
| 클라이언트 수 | 50/100명, join 0.5, 배치 32 | 2 × 3 seed | 예전 원고 표 1 재사용 |
| 로컬 에폭 | E = 2/5/10 | 3 × 3 seed | 예전 원고 표 3 재사용 |
| label skew 강도 | Dir α = 0.3, 1.0 | 2 × 3 seed | 새로: Local, FedTGP, FedSA, FedAPEN-ph |

E=10은 라운드당 시간이 10배라 seed 1개로 먼저 본다. 게이트 patience는 라운드 단위라 E가 크면 학습량 기준으로 길어진다는 점을 본문에 밝힌다.

## 6. 진단 실험 (분석 그림, 대부분 완료)

| 진단 | 근거 run | 남은 일 |
|---|---|---|
| F1 구조 간 채널 = Local | 기준선 표 + Local | Local 500R |
| F2 같은 구조 평균만 지식 전달 | 평균 vs 비평균 쌍둥이 (+11) | 500R seed 3개 (4절) |
| F3 head 평균 붕괴, label prior 원인 | H3 실험 (BSM으로 17 → 55.7) | 완료 |
| F4 쌍둥이 암기 (train−held-out 차이) | TwinGate 로그 | 그림 |
| F5 평균 후 head 표류 → 재정렬 | cRA vs 대조 | 완료 |
| 보완성 분석 (router 불가) | router2.py: 샘플별 oracle +6.3, 상한 router도 고정 α 못 넘음 | 그림 |

## 7. 효율과 실기기

| 항목 | 방법 | 상태 |
|---|---|---|
| 통신량 | 방법별 라운드당 payload × 동결 라운드 (계산). c8은 R234 동결로 FedTGP 500R 총량보다 적음 | 계산식 정리 |
| 학습 연산 | MACs × private 학습 비율(priv_frac 로그) | 쌍둥이 게이트 seed 3개 후 |
| 추론 연산 | cascade.py (쌍둥이 확신 < τ일 때만 private) | τ를 held-out으로 고르게 수정 |
| 기기 측정 | `system/device_bench.py`: Raspberry Pi(CNN, MobileNetV2), Jetson Nano(GoogLeNet, ResNet-18/34/50), 워크로드별 step 시간·에너지 | 사용자가 기기에서 실행 |
| 500R 누적 기기 비용 | 동결 전 라운드 × ours_pre + 동결 후 × ours_cache(게이트 반영) vs Local·FedAPEN·FedTGP | 기기 결과 받은 뒤 |

## 8. 범위와 한계 (본문에 정직하게)

- 증강 보조 표: CIFAR-100에서 증강을 켠 우리 방법, Local, FedSA (seed 1개). 다른 기준선은 증강 옵션이 없어 코드 작업이 필요하다.
- 센서(PAMAP2T): 평균 효과 없음. 데이터 부족이 아니라 시간에 따른 분포 이동이 문제인 경우는 범위 밖.
- private 기여가 작다(+0.5~0.7). 결합 이득의 대부분은 쌍둥이에서 온다.

## 9. 실행 순서와 예산

| 단계 | 내용 | run 수 (우리) | 기간 (어림) |
|---|---|---|---|
| 0 | 최종 구성 확정 | 진행 중 | ~11-19 오전 |
| 1 | 본 표 우리 9 run + 다른 노드: Local, FedAPEN-ph, FedMRL | 9 | Tiny가 병목, 약 1일 |
| 2 | 핵심 ablation seed 3개 | 약 15 | 약 1일 |
| 3 | 일반화 | 약 40 | 2~3일 |
| 4 | 효율·기기·증강·진단 그림 | 소수 | 병행 |

GPU 5개 × 2 슬롯 기준. CIFAR 500R은 GPU를 나눠 쓰면 3~4시간, Flowers 약 1.5시간, Tiny는 혼자 써도 약 9시간.
