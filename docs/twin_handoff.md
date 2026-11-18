# 쌍둥이 잔차 방법: 인수인계 (2026-11-18)

다른 노드나 새 세션이 이어서 작업할 수 있도록 현재 상태, 결정 사항, 실험 규칙을 정리한 문서다.
답변은 한국어로, 연구 동료처럼(아이디어의 약점과 반례를 먼저) 한다.

## 1. 연구 한 줄 요약

여러 제조사 IoT 기기(용량 등급이 다름, 증강 불가, 극단적 label skew)의 이기종 FL.
모든 기기가 **같은 구조의 작은 쌍둥이(CIFAR ResNet-20, 0.27M)** 를 하나 더 두고 backbone만 FedAvg 하며,
각자의 이기종 private 모델은 쌍둥이 위에 **잔차**로 얹는다(예측 = 쌍둥이 logit + private logit).
쌍둥이가 클라이언트 데이터 전체를 암기하기 시작하면 **게이트**가 쌍둥이를 고정하고 공유(통신)를 끊는다.

## 2. 진단 (논문의 핵심 기여 후보)

| # | 발견 | 근거 |
|---|---|---|
| F1 | 증강이 없으면 교차 구조 채널(프로토타입, logit, 생성기)은 Local과 차이가 없다 | 기준선 표 (Local 500R은 아직 필요) |
| F2 | 같은 구조의 가중치 평균만 지식을 옮긴다 | 평균한 쌍둥이 vs 평균하지 않은 쌍둥이: +11 (500R) |
| F3 | 공유+private 계열(FedAPEN 등)은 공유 모델 head까지 평균한다. label skew에서는 이 때문에 공유 모델이 개인화 평가에서 무너진다 | head까지 평균하면 쌍둥이 단독 17.8% (A1). 수정판 FedAPEN 공유 모델도 R10에 6.8% |
| F4 | 평균되는 쌍둥이는 결국 클라이언트 데이터 전체를 암기한다 | 게이트의 train−held-out 차이가 0에서 9~11까지 커짐 |

적용 범위: PAMAP2T(시간순 분할 센서 데이터)에서는 평균 효과가 없었다(평균 86.36 vs 비평균 86.88).
그 데이터의 문제는 데이터 부족으로 인한 암기가 아니라 한 사람 안에서 시간에 따라 분포가 바뀌는 것이다.
또 PAMAP2T에서는 게이트용 held-out을 무작위로 뽑으면 50% 겹치는 window 때문에 정보가 샌다. 시계열은 구간의 마지막 5%를 써야 한다.

## 3. 설계 결정 (실험 근거)

| 결정 | 근거 (CIFAR-100, 150R, seed 0, R121~150 평균, 합) |
|---|---|
| **앵커 제거, 쌍둥이 lr ×2** (`--fedsa_cm_lr_scale 2`) | 앵커 없음 52.06(2회 평균) → 앵커 53.52(2회) → 앵커 없음+lr×1.5 54.80 → **앵커 없음+lr×2 55.98**. 앵커 효과는 업데이트 보폭이었다 |
| 쌍둥이 head는 개인 (`--fedsa_cm_base_only true`) | head도 평균하면 46.06 (−5.9) |
| private 쪽 앵커, 마진, λ, CC 없음 | 모두 ±0.5 이내 (잡음) |
| 단방향 학습 (쌍둥이는 private에서 배우지 않음) | 양방향 54.91 vs 단방향 54.98. 단순해서 단방향 유지. "암기 오염 방지"는 기각된 가설이라 기여로 쓰지 않는다 |
| 고정 랜덤 head, FedNTD는 쓰지 않음 | 47.3 / 49.57로 실패 |
| 같은 seed 재실행 잡음 | 0.3~0.8. 1 미만 차이는 판단하지 않는다 |

**진행 중인 판단 (결과가 나오면 이 문서 갱신):**
- S1b/S2b: 잔차 합 CE와 private KL을 빼도 되는지 (`--fedsa_res_sum_loss false`, `--fedsa_res_kl 0`)
- 암기 늦추기: A 서버 보폭(`--fedsa_cm_server_lr 2`), B 부호 일치 평균(`--fedsa_cm_agree 0.2/0.4`), D private KL ×2(`--fedsa_res_kl 2`)
- lr ×3, FedAPEN 공유 lr ×2(공정성, `--apen_shared_lr_scale 2`)
- 게이트 효과: 예전 구성으로 500R 게이트 끈 run 진행 중. 최종 구성에서 다시 재야 함
- 데이터 없는 게이트: 클라이언트 업데이트 일관성(`TwinCoherence rho`, 로그에 매 라운드 기록)이 held-out 최고점과 맞는지 (D3)

**잠정 최종 구성 명령 (500R):**

```
python -u main.py -did 0 -dev cuda -data Cifar100 -ncl 100 -m HtFE-img-8 -fd 512 -lbs 128 -lr 0.01 -gr 500 -eg 10 -ls 1 -jr 1 -nc 20 \
  --use_amp false --pin_memory true --fedsa_augment false --random_seed 0 \
  -algo FedSA --fedsa_anchor_momentum 1.0 --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min \
  --fedsa_common_model true --fedsa_common_arch c20 --fedsa_cm_base_only true --fedsa_kd_frozen true \
  --fedsa_residual true --fedsa_kd_weight 0 --fedsa_cm_val 0.05 -lam 0 -mcl 0 -cc 0 --fedsa_cm_gate 20 --fedsa_res_standalone true \
  --fedsa_cm_lr_scale 2 -go <이름> -sfn temp_<이름>
```

할 일: 앵커를 쓰지 않으면 프로토타입 교환도 필요 없다. 이 교환을 끄면 게이트 고정 뒤 통신이 0이 된다. 아직 코드에서 끄지 않았다.

## 4. 기준선

**사용자 측정 (증강 없음, HtFE-img-8, 500R, seed 3개, 10라운드마다 평가, seed 평균 곡선의 최고값):**

| 방법 | CIFAR-100 | Flowers102 | Tiny-ImageNet |
|---|---|---|---|
| LG-FedAvg | 40.64 | 53.80 | 23.64 |
| FedGH | 39.83 | 53.06 | 23.19 |
| FedProto | 38.24 | 49.88 | 18.24 |
| FedTGP | 44.51 | 55.60 | 23.20 |
| FedSA (재측정, 확정) | 45.02 | 55.75 | 23.88 |
| FedRG | 41.98 | 50.62 | 24.15 |

"FedSA 45.88"은 예전 Proposed 값이다. 쓰지 않는다.

**우리 방법 500R (예전 구성: 앵커 λ10·MCL·CC, lr ×1. 최종 구성으로 다시 돌릴 예정):** CIFAR-100 60.42±0.31, Flowers102 64.25±1.55.

**버그 경고 (커밋 c043724에서 수정):** `-sfn temp_*` 메모리 저장소에서 `load_item('Server', ...)`가 복사본이 아니라 서버 객체 자체를 반환했다.
그래서 **FedAPEN, FML, FedMRL, FedKD** 는 FedAvg가 아니라 20개 클라이언트가 공유 모델 하나를 순차 학습했다. 이 커밋 이전 결과는 무효다
(예: FedAPEN-c20 500R 52.28). LG-FedAvg, FedGH, FedGen(값 복제), 프로토타입 계열(읽기만), 우리 쌍둥이는 영향이 없다.

**다른 노드에서 돌릴 기준선 (500R × seed 0,1,2):**

| 기준선 | 데이터셋 | 비고 |
|---|---|---|
| FedAPEN | Flowers102, Tiny-ImageNet (CIFAR은 이 노드에서 진행 중) | `-algo FedAPEN --global_arch c20` (공유 모델을 우리 쌍둥이와 같은 구조로) |
| FedMRL | CIFAR-100, Flowers102, Tiny-ImageNet | `-algo FedMRL --global_arch c20` 권장. c20 조합은 아직 안 돌려 봤으니 3라운드 smoke 먼저 |
| Local | CIFAR-100 | 진단 F1용. 이 노드에서는 FedSA에 앵커 손실을 모두 끈 형태(`-lam 0 -mcl 0 -cc 0`)로 썼다 |

공통 인자: `-m HtFE-img-8 -fd 512 -lr 0.01 -gr 500 -eg 10 -ls 1 -jr 1 -nc 20 --use_amp false --pin_memory true --fedsa_augment false`.
데이터셋별: CIFAR-100 `-data Cifar100 -ncl 100 -lbs 128`, Flowers102 `-data Flowers102 -ncl 102 -lbs 32`, Tiny `-data TinyImagenet -ncl 200 -lbs 128`.
제외하기로 한 것: FML, FedKD, pFedAFM/pFedES 구현, PAMAP2T 기준선 재측정, 증강 보조 표.

## 5. 남은 실험 (우선순위 순)

1. 최종 구성 확정 (위 진행 중 판단)
2. 본 표: 우리 방법 × 3 데이터셋 × 3 seed + 신규 기준선
3. 핵심 ablation 500R × 3 seed: 평균 끔, head 평균, 게이트 끔, lr ×1
4. 모델 이질성 HtFE-img-2/3/4/5/9 (CIFAR-100)
5. 클라이언트 수 50/100 (q=0.5, 배치 32), 부분 참여 100명 q=0.1/0.3
6. 로컬 에폭: **에폭 × 라운드 = 1000** (E=2는 500R, E=5는 200R, E=10은 100R). 게이트 patience는 라운드 단위라 E가 클수록 학습량 기준으로 길어진다
7. 비용 표: 라운드당 시간(GPU를 혼자 쓸 때), 100라운드 업로드·다운로드

## 6. 실행 규칙

- `PYTHONPATH=stylegan/stylegan-utils` 필요. `main.py`가 CPU 스레드를 4개로 제한한다(`TORCH_NUM_THREADS`).
- `-sfn temp_<run 이름>`만 쓴다(`-sfn temp` 금지). temp_는 메모리 저장소라 끝난 run의 모델은 다시 불러올 수 없다.
- GPU 하나에 run 2개까지, GPU 여유 메모리 4GB 이상, 디스크 5GB 이상. Tiny-ImageNet은 GPU 하나에 하나만(약 5.6~6.5GB, 둘이면 OOM).
- 대기열 스크립트 여러 개가 동시에 자리를 고르면 한 GPU에 몰려 OOM이 난다. 대기열은 하나로 합치거나 잠금을 쓴다.
- 이미 run을 띄운 백그라운드 대기열을 강제로 멈추면 그 run까지 죽는다. 멈추지 말고 `logs/<이름>.launch.out` 자리표시 파일로 건너뛰게 한다.
- `pgrep -f`/`kill` 패턴이 자기 셸 명령과도 일치할 수 있다. `grep "[p]ython -u main.py"` 식으로 쓰고, PID를 먼저 출력해 확인한다.
- 결과 집계: 사용자 프로토콜은 "라운드별 seed 평균 곡선의 최고값". 우리 방법의 예측 정확도는 로그의 `Fused ... logit_sum=` 값이다.
