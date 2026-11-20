# 논문 스토리와 실험 구성 (2026-11-19 밤 기준)

목표: **IEEE TMC**(11-20 결정, 안 되면 IoT-J). TMC는 실기기 평가와 이유 분석을 IoT-J보다 더 요구한다. 실험 규칙·프로토콜은 `docs/experiment_plan.md`, 진행 기록은 `system/logs/night_actions_20261117.md`, 기준선 측정은 `docs/baselines_progress.md`.

## 0. 제목 (11-20 사용자 결정)

**Rethinking Knowledge Sharing in Model-Heterogeneous Federated Learning: Share a Tiny Common Model, Specialize Locally**

- 방법 이름 FedScout는 제목에 넣지 않고 초록·본문에서 소개한다("scout"는 처음 보는 독자에게 뜻이 안 통함). IoT는 초록 첫 문장과 키워드로 드러낸다.
- "Specialize Locally"는 2단계 설계(연합 뒤 로컬 학습)가 통과하면 더 정확해진다.

## 진행 요약 (11-20 저녁 기준, 이 문서의 결정과 남은 계획 한눈에)

**확정된 것**
| 항목 | 결정 | 근거·위치 |
|---|---|---|
| 제목 | Rethinking Knowledge Sharing in Model-Heterogeneous Federated Learning: Share a Tiny Common Model, Specialize Locally | 0절 |
| 투고처 | **ICML 2027 먼저(초록 2027-01-14, 본문 01-22 AoE), 떨어지면 TMC**(11-20 밤 사용자). 사용자 2027-08 졸업, 학위논문은 이 연구로 | 아래 ICML 일정 |
| 기여 | 진단(어떤 통로로도 private이 공유 + log 비율을 넘지 못함) + 그 진단이 정당화하는 설계 + 효율 | 7절, `docs/related_work.md` |
| 이론 | 명제 1(label shift: 공유 + log 비율 = 클라이언트별 Bayes 최적), 명제 2(feature shift에서만 private이 필요) | `docs/theory_label_shift.md` |
| 최종 방법 | 2단계 FedScout: c8 scout(projector 없음) 전체 FedAvg, head lr ×8, R490 동결·개인화, learner는 동결 뒤에만 학습, **로컬 단계 E = 50 epoch(`-gr 540`, 11-20 밤 확정)**, screening τ 0.5, cascade | 4절. Flowers 69.62(매 라운드 learner 70.15), 학습 연산 FedAPEN 대비 5.8배 적음. E 근거: 10/50/100 → 69.32/69.62/69.52, 기여 −0.28/+0.91/+0.81 |
| 보고 규칙 | 동결 뒤 로컬 학습은 통신 0이므로 "490R 연합 + E 로컬 epoch"로 따로 표시(그림은 점선) | 8절 |
| 기준선 | 원 논문 lr. 공유 모델 방식은 c8 | 메모리 |
| 기기 | **Raspberry Pi 5(CPU 학습·추론), Jetson Orin Nano Super(GPU 학습·추론)** — 둘 다 측정 완료. Jetson Nano 제외(4GB + 다른 계정 데스크톱으로 r34부터 스왑), Hailo 생략(Dataflow Compiler가 연구실에 없음; 11-20 사용자) | 기기 실측 절 |

**진행 중 (11-20 저녁)**
| 실험 | 판정 | 예상 |
|---|---|---|
| 이유 분석 B·C·D: FedAPEN `an_*` 6개(클래스 빈도 3분위, Dir α 0.3/1.0, IID, feature shift, 보정 강도 τ) | 8절 사전 등록 기준 | 20~21시 |
| feature shift에서 2단계 FedScout(`lafsa8_*fs`)와 매 라운드 learner 상한(`npsa8_*fs`) | 데이터 양 vs label shift 설명 | 18시 / 21시 |
| Flowers 동결 뒤 로컬 50·100 epoch(`lafsa8L*`) | 로컬 epoch 수 E 결정 | 19시 반 |
| 기기: Pi 5 완료, Orin 완료, Jetson Nano 제외 | 기기 표 완성 | 완료 |

**ICML 2027 일정 (본문 마감 2027-01-22까지 약 9주)**
| 기간 | 할 일 |
|---|---|
| 11-21 ~ 11-27 | 본 결과 완성(63: Tiny, ablation, 일반화 / ariel: 기준선 11-24), 이유 분석 seed 3개, **실제 feature shift 데이터 준비**(DomainNet 10클래스 부분집합(FedBN 방식) 또는 Office-Caltech10, Digit5 중 선택) |
| 11-28 ~ 12-04 | **진단 확대**: FML·FedMRL·pFedMoE에 공유 + 비율 진단 추가(Dir 0.1, feature shift). 실제 feature shift에서 FedScout·FedAPEN 진단 |
| 12-05 ~ 12-18 | **"언제 private이 이기나" 정량화**: 공유 모델 품질(R330 → R500에서 순위가 뒤집힘)과 클라이언트당 데이터 양(50·100명 결과)의 함수로. 그림·표 확정 |
| 12-19 ~ 01-08 | ICML 형식 원고(**본문 8쪽** + 참고문헌·부록) 초안, 모의 심사(academic-paper-reviewer)로 수정 |
| 01-14 / 01-22 | 초록 등록 / 본문 제출 |
- ICML은 상호 심사(저자 중 심사 자격자 의무)와 제출 수 제한이 있다. 초록 등록 전에 확인한다.

**ariel 기준선은 11-24 무렵 완료 예정(11-20 사용자).** 그동안 63 노드 GPU 5장으로 우리 실험을 한다(아래 순서).

**63 노드 계획 (11-20 저녁 작성, 지금 큐가 빠지는 11-21 오후부터)**
| 순서 | 실험 | run 수 | 비고 |
|---|---|---|---|
| 0 | 로컬 epoch E 확정 | — | **E = 50 확정(11-20 밤)**. 아래 run은 모두 `-gr 540`, 큐 `queue_final.py`(36개: CIFAR 본 표 3, ablation 12, 일반화 18, Tiny 3)가 B·C 큐 뒤에 대기 |
| 1 | 본 표 최종 구성: Tiny × 3, CIFAR × 3(E가 10이 아니면) | 3~6 | Tiny는 GPU 단독 |
| 2 | ablation (CIFAR × 3): 평균 없음, 개인 head(backbone만 공유), head lr ×1, 로컬 단계 screening 끔 | 12 | scout 단독(= c8 FedAvg + head FT)과 매 라운드 learner는 기존 로그로 대체 |
| 3 | 일반화 (CIFAR × 3): HtFE 2/3/4/5/9, lowend | 18 | 기준선은 예전 원고 표 1을 그대로 쓴다(11-20 사용자: 지금과 같은 설정으로 돌린 값) |
| 4 | 그림·표: 그림 3종 OURS 교체(로컬 구간 점선), 효율 표(FLOPs + Pi/Orin 실측), 이유 분석 표(seed 3개), 상한 분석 그림, τ 그림 | — | 계산만 |
| 5 | 로컬 에폭 (부록, 맨 마지막) | 9 | |

1. E 결정 → 최종 구성으로 본 표 Tiny × 3 seed, ablation 재정의(평균 없음, 동결 시점, 개인 head, head lr ×1), 일반화(HtFE 2/3/4/5/9, lowend), 클라이언트 수(ariel a2019102224 `METHOD=FedScout_sa`), 로컬 에폭은 맨 마지막.
2. 그림 3종의 OURS를 `lafsa8`로 교체, 로컬 구간 점선, 효율 표(FLOPs + 기기 시간·에너지·가장 느린 클라이언트).
3. 이유 분석 결과로 상한 분석 그림("어떤 통로도 공유 + 비율을 넘지 못함")과 τ 그림.
4. J7-C 전력계로 벽 전력 재측정(현재 블루투스 연결 안 됨; 연구실 jhs님이 같은 측정기로 전력을 잰 흔적이 있어 켜는 방법 확인). Hailo는 생략(11-20).
5. 가장 가까운 논문 본문 정독(PersFL, ProxyFL, FedType, FedRoD, Xu & Huang, FedTopo) → 원고 작성.

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

**11-20 저녁 확정 (사용자): 2단계 FedScout.** c8 scout(projector 없음, 원래 분류기 64×C) 전체(head 포함)를 FedAvg, head lr ×8(`--scout_head_lr_scale 8`), R490에 동결·개인화, 이기종 learner는 **동결 뒤에만** 학습(`--scout_learner_after_freeze`), P2 잔차 + held-out α, screening τ=0.5, cascade 추론, realign 0, 평가 때 head 1 epoch 미세 조정(`--scout_eval_ft`). scout lr 0.03·private lr 0.01. run 이름 `lafsa8_*`, Slurm `METHOD=FedScout_sa`.
- 근거(Flowers, seed 3개 평균 곡선 최고): 69.32 ± 1.27, 클라이언트당 학습 연산 44 TFLOPs. 매 라운드 learner를 학습하는 shareall(npsa8) 70.15 ± 1.25·257 TFLOPs, FedAPEN native 290 TFLOPs, private만 매 라운드 학습하는 기준선 약 245 TFLOPs. 즉 **FedAPEN 대비 6.6배, private만 학습하는 기준선 대비 약 5.6배 적은 연산.** 연합 중 라운드 시간도 약 4초(npsa8 15~18초, GPU당 2 run).
- 아래 문단은 11-19 구성(기록용).

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
| **기기 실측** | Raspberry Pi 5(CPU, 전력은 PMIC 레일 합), Jetson Orin Nano Super(GPU, tegrastats VDD_IN). 학습: scout 한 step(연합 단계), 동결 뒤 learner 한 step(screening 30%/100%), 매 라운드 learner, Local, FedAPEN. 추론: scout, private, cascade(= scout + 미확신 비율 × private). 전체 기기 시간 = epoch 수 × (n/B) × step 시간으로 조합 | **완료(11-20, Claude 원격 실행)**: `docs/figs/data/device/`, `system/device_compose.py`. Jetson Nano 제외, Hailo 생략. 벽 전력(J7-C)은 미측정 |
| **분석 표 (관찰 2)** | FedAPEN: 앙상블 / private / 공유 / 공유 + 비율. FedScout: 결합 / scout 단독 / learner | 로그로 작성 가능(run 불필요). Tiny FedAPEN: 공유 + 비율 45.7 > 앙상블 39.1, FedScout 결합 = scout |
| **ablation (CIFAR × 3 seed)** | screening 끔 / 재정렬 끔 / I1 learner / head까지 평균(shareall, R490에 개인화) / 평균 없음 / scout lr×1 | 대기열(우선순위 3번째) |
| 일반화: 클라이언트 수·참여율 | 50명 join 0.5, 100명 join 0.5·0.3·0.1 × 3 seed | 대기열. 기준선은 ariel이 같은 설정으로 측정 중 |
| 일반화: 모델 이질성 | HtFE 2·3·4·5·9, lowend × 3 seed | 11-21 오후 큐 예정. 기준선은 예전 원고 표 1 재사용(같은 설정으로 돌린 값, 11-20 사용자 확인) |

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

## 7. 선행 연구와 노벨티 (전체 목록과 URL·읽은 정도: `docs/related_work.md`)

### 요약 (11-20 검색 3갈래, 대부분 초록 수준; 본문 확인: PersFL, Yu 외 2020, ProxyFL, FedType, FedRoD, pFedMoE, FedMRL, FedLC, ReBaFL)

- **새로움(찾지 못함)**: 학습 없는 "공유 + log 지역 비율"이 FML/FedAPEN 계열 앙상블과 같거나 낫다는 이기종 FL 진단과, 이득의 공유·private·분포 분해. pFedMoE·FedMRL은 분해 없이 Standalone 대비 +0.6~2.9만 보고.
- **새로움(조합)**: 이기종 private을 연합 중 학습하지 않고 끝난 뒤 한 번만 학습 + 기기 안 cascade(작은 모델 → private).
- **새롭지 않음(인용 필수)**: 사후 log-prior 보정(Menon et al. ICLR 2021, Balanced Softmax NeurIPS 2020), FL prior 보정(FedLC ICML 2022, FedRS KDD 2021, ReBaFL), 개인 head ≈ 클래스 분포(FedRoD ICLR 2022, FedTHE ICLR 2023), 균형 전역 + prior 보정 개인화(Xu & Huang 2023), FedAvg 후 로컬 증류·FT 1회(PersFL 2021, Yu 외 2020; 같은 구조만), 작은 공유 + 이기종 private(ProxyFL, FML, FedKD, FedType, FedMRL; 모두 매 라운드 private 학습), 기기는 작은 모델만(FedGKT, 큰 모델은 서버), 확신도 cascade(SplitGP, FedHLM, Dirichlet-UQ PFL), 샘플 선택(FedBalancer).
- **예상 공격**: "그냥 Bayes label-shift 보정", "PersFL의 학생을 이기종으로 바꾼 것", "FedAvg-FT와 무엇이 다른가". → 기여를 기법이 아니라 진단 + 진단이 정당화하는 설계 + 효율로 건다.
- **추가할 것**: c8 FedAvg + head FT(private 없음) 줄(scout 단독), 연합 뒤 private 학습 vs Local 단독 private, ProxyFL/FedKD 기준선 검토, 가장 가까운 논문 본문 정독.

## 8. 이유 분석 (11-20 사용자: "뭘 하던 이유 및 분석은 필요", A~D 진행, E 제외)

### A. 설명 (가설): label shift에서는 공유 + 비율이 최적 개인화다
- **정식 초안(명제 1·2, 증명, 따름정리, 한계, 확인한 인용): `docs/theory_label_shift.md`.** 아래는 요약.
- Dirichlet label skew는 label shift다: p_k(x|y) = p(x|y)는 모든 클라이언트가 같고 p_k(y)만 다르다.
- 그러면 클라이언트 k의 Bayes 최적 분류기는 p_k(y|x) ∝ p(y|x) · p_k(y) / p(y). logit으로는 f(x) + log p_k(y) − log p(y) (Saerens et al. 2002, Lipton et al. 2018 BBSE, Menon et al. 2021). CIFAR-100은 전역 비율이 균등이라 log p(y)는 상수.
- 즉 이 설정에서 개인화의 상한은 "잘 학습된 전역 모델 + log 지역 비율"이고, 이기종 private이 그 이상 줄 정보는 원리상 없다. private이 따로 배우는 나머지는 적은 지역 데이터의 과적합이다.
- 예측: label shift에서는 공유 + 비율 ≥ 앙상블, feature shift(p_k(x|y)가 다름)에서는 private이 비율 이상을 더한다(앙상블 > 공유 + 비율).
- 인용 확인함: Saerens 2002(Neural Computation 14(1)), Lipton 2018(ICML, PMLR 80).

### B~D 실험 (사전 등록, 결과 보기 전 작성)
FedAPEN c8 native head(본 표 버전), seed 0, 500R, 나머지 프로토콜 동일, `--apen_dump_perclass true`(마지막 평가의 클라이언트별·클래스별 정답 수를 `logs/<이름>_perclass.json`에 저장). run 이름 `an_<데이터>_s0`.

| # | 데이터 (`dataset/`) | 무엇을 보나 | 맞으면 | 폐기 기준 |
|---|---|---|---|---|
| B | `Cifar100` (Dir 0.1) | 클래스를 지역 학습 빈도 3분위(다수/중간/소수)로 나눠 private·공유·공유 + 비율·앙상블 비교 | private의 우위는 다수 클래스에 몰림. 소수 클래스에서 공유 + 비율 ≥ private | 소수 클래스에서 private이 공유 + 비율보다 1점 넘게 높음 |
| C | `Cifar100_a0.3`, `Cifar100_a1` (Dir 0.3/1.0, 같은 생성기, numpy seed 1) | skew 강도별 진단 | 두 α 모두 공유 + 비율 ≥ 앙상블 − 1. 비율 이득(공유 + 비율 − 공유)은 α가 클수록 작아짐 | α=1.0에서 앙상블이 공유 + 비율보다 1점 넘게 높음 |
| D | `Cifar100_iid_fs` (IID 라벨 + feature shift) | 결정적 실험 | 앙상블 > 공유 + 비율 + 1 (private이 도메인 정보를 더함) | 앙상블 ≤ 공유 + 비율 + 1이면 label shift 설명은 틀림(→ 데이터 부족 등 다른 원인) |
| D-대조 | `Cifar100_iid` (IID 라벨, shift 없음) | private의 이득이 용량·앙상블 효과인지 | 앙상블 − 공유 + 비율 ≤ 1 | 여기서도 앙상블이 1점 넘게 높으면 D의 이득은 도메인 특화가 아니라 용량 |
| D-혼합 | `Cifar100_fs` (Dir 0.1 + feature shift) | 두 이동이 함께 있을 때 | 앙상블 > 공유 + 비율 + 1 | — |

- feature shift는 `dataset/generate_cifar100_fs.py`: 클라이언트 i는 도메인 i % 5(원본, 흑백, 반전, 90도 회전, 16×16 픽셀화)로 이미지를 본다. 라벨 분할은 원본과 같고 변환은 라벨과 무관하다. DomainNet은 이미지 18GB 이상 다운로드, Digit5는 원본 .mat가 없어 통제된 합성 shift를 먼저 쓴다(실제 데이터는 필요하면 부록).
- 한 seed로 먼저 보고, 차이가 폐기 기준 근처(±1)면 seed 3개로 늘린다.

### 2단계 설계 검증 (같은 날 시작)
- `lafsa8_<데이터>_s<0-2>`: npsa8(shareall, projector 없음, head lr ×8, freeze_at 490) + `--scout_learner_after_freeze`. learner는 R490 이후 10라운드만 학습.
- 폐기 기준(사전): seed 평균 곡선 최고가 npsa8보다 0.5 넘게 낮거나, FedAPEN 대비 학습 연산 절감이 3배 미만.
- **결과(Flowers)**: 69.32 ± 1.27 vs npsa8 70.15 ± 1.25 (−0.83, 정확도 기준 미달), 연산 FedAPEN 대비 6.6배 절감(통과). ~~차이는 scout 궤적의 seed 변동~~ **정정(17시)**: seed 0만 보고 내린 잘못된 해석이었다. R480의 −0.58은 lafsa8(learner 없음)과 npsa8(결합)의 차이라 learner 기여 자체다. seed 3개 R500 learner 기여(결합 − scout): npsa8 Flowers +1.46/+0.63/+1.75(평균 +1.28), CIFAR +0.52/+0.71/+0.60(+0.61) vs lafsa8 Flowers 0/+0.24/−1.07(−0.28), CIFAR +0.26/+0.73/+0.50(+0.50). scout 단독은 두 구성이 거의 같다(Flowers R500 평균 68.69 vs 68.48). 즉 Flowers의 −0.83은 learner 10라운드가 부족해서 생긴 차이다(CIFAR는 10라운드로 충분). **사용자가 정확도 기준 미달을 받아들이고 채택(11-20, "연산이 겁나 적잖아. 이것만 봐도 기여다").** 논문에는 두 구성의 정확도·연산을 모두 보고한다.
- **결과(CIFAR, 같은 날 17시)**: 66.05 ± 0.91 vs npsa8 66.23 ± 0.78 (−0.18, **정확도 기준 통과**), 학습 연산 80.9 vs FedAPEN native 544.2 TFLOPs/client(**6.7배**), npsa8 496.9. R500 learner 기여(결합 − scout, seed 0/1/2): CIFAR +0.26/+0.73/+0.50(α 0.25, learner 단독 4~5%), Flowers 0/+0.24/−1.07(α 0/0.1/0.25; s2는 held-out으로 고른 α가 test에서 손해). cascade(τ 0.5)는 private을 샘플의 30~35%에서만 실행.
- head lr 결정 근거: projector 없음 shareall, head lr ×1 → ×8이 Flowers 64.53 → 70.15(500R, seed 3개 평균 곡선 최고), CIFAR는 R400에 이미 65.12(×1 최종 62.75).

### 2단계 FedScout on feature shift (11-20 저녁, 사전 등록)
- run: `lafsa8_<데이터>_s0`(learner 10라운드, 최종 구성), `npsa8_<데이터>_s0`(learner 매 라운드, 상한), 데이터 `Cifar100_iid_fs`, `Cifar100_fs`. 지표: 마지막 평가의 결합값 − scout 단독값(ScoutEval fused − scout), 고른 α.
- 경쟁 가설: **데이터 양**(사용자: scout는 전체 분포를 보고 private은 자기 데이터만 본다 → 어떤 이동에서도 기여 ≈ 0) vs **label shift**(명제 2: feature shift에서는 private이 기여).
- 예측: label shift 설명이 맞으면 npsa8(상한)에서 결합 − scout > +1. 데이터 양 설명이 맞으면 npsa8에서도 ≤ +1. lafsa8에서 기여가 상한보다 1점 넘게 작으면 learner 10라운드가 부족 → 동결 시점을 앞당기는 변형 검토.

### 연합 뒤 로컬 단계 (11-20 저녁, 사용자: "동결 뒤 로컬 학습은 따로 표시")
- **표기 규칙**: 동결(R490) 뒤 라운드는 통신이 0이므로 연합 라운드로 세지 않는다. 표에는 "490R 연합 + E 로컬 epoch"로 쓰고, 기준선의 500R와 나란히 둔다.
  - 라운드 축 그림: R490까지는 실선, 그 뒤 로컬 epoch 구간은 다른 선 모양(점선)과 별도 눈금("+E local")으로 그린다.
  - 통신 축 그림: 로컬 구간은 통신이 늘지 않으므로 곡선 끝의 수직 구간으로 나타난다. 그대로 두고 주석을 단다.
  - 연산 축 그림과 효율 표: 로컬 구간의 learner 학습 연산을 모두 포함한다.
  - 지금 최종 구성(lafsa8)도 마지막 10라운드는 로컬 단계다 → "490R 연합 + 10 로컬 epoch"로 표기한다.
- **실험**: `lafsa8L{50,100}_Flowers102_s{0-2}` = lafsa8 + `-gr 540/590` (동결은 그대로 R490, 동결 뒤 50/100 로컬 epoch).
- 예측(사전): learner 기여가 npsa8(+1.28)에 가까워지고, 최종 정확도가 npsa8(70.15)과 0.5 이내. 폐기 기준: 100 epoch에서도 기여 ≤ +0.3이면 step 수가 아니라 연합 중 공동 학습이 중요하다는 뜻.


### 기기 실측 1차 (11-20 17시 반, 연구실 IoT 기기, 측정은 Claude가 원격 실행)
- 기기: Jetson Orin Nano Super(jetson3, GPU, tegrastats VDD_IN = 보드 입력 전력), Raspberry Pi 5(master, CPU 4코어, 전력은 PMIC 레일 합 = SoC + 메모리, 벽 전력의 하한). 배치 32, `system/device_bench.py`(최종 구성), 원자료 `docs/figs/data/device/*.csv`, 조합 `system/device_compose.py`.
- 조합: 클라이언트 i는 HtFE-img-8의 i % 8 모델, 지역 데이터 수는 실제 분할(`flops.json` n_train), 500R·1 epoch. FedScout 2단계 = 490 × scout 학습 + 10 × 동결 뒤 learner(screening 30%).

| 기기·데이터 | FedScout 2단계 | 매 라운드 learner | FedAPEN | private만 학습 (LG/Proto/TGP/SA) |
|---|---|---|---|---|
| Orin, CIFAR: 클라이언트 평균 시간 / 가장 느린 클라이언트 / 평균 에너지 | 0.18 h / 0.22 h / 4.8 kJ | 0.96 h / 2.68 h / 31.8 kJ | 0.98 h / 2.62 h / 32.4 kJ | 0.82 h / 2.46 h / 27.4 kJ |
| Orin, Flowers | 0.04 h / 0.04 h / 1.5 kJ | 0.14 h / 0.38 h / 7.5 kJ | 0.14 h / 0.38 h / 7.7 kJ | 0.11 h / 0.35 h / 5.9 kJ |
| Pi 5, CIFAR (8개 모델 전부, 18시 완료) | 2.21 h / 2.93 h / 40.8 kJ | 27.21 h / 91.34 h / 528 kJ | 27.80 h / 90.85 h / 589 kJ | 25.85 h / 90.91 h / 562 kJ |
| Pi 5, Flowers | 1.64 h / 1.90 h / 29.6 kJ | 6.35 h / 16.38 h / 130 kJ | 6.38 h / 16.95 h / 137 kJ | 4.84 h / 15.20 h / 105 kJ |

- 읽는 법: 평균 시간 기준 FedAPEN 대비 Orin 5.4배(CIFAR)·3.8배(Flowers), Pi 12.6배(CIFAR)·3.9배(Flowers). Pi CIFAR의 가장 느린 클라이언트(r152)는 FedAPEN 90.9 h vs FedScout 2.9 h(31배). **가장 느린 클라이언트 기준(연합 라운드 시간을 정하는 낙오자)은 약 12배**: 연합 단계에서는 모든 기기가 같은 c8만 학습하므로 모델 이질성이 라운드 시간에 영향을 주지 않는다.
- GPU(Orin)에서는 배치 32가 작아 screening(30%)이 learner 한 step 시간을 거의 줄이지 못한다(r18 42.5 ms vs 전체 37.0 ms). 절감은 연합 중 learner를 학습하지 않는 데서 나온다. CPU(Pi)에서는 screening이 step 시간을 약 절반으로 줄인다(r18 729 vs 1495 ms).
- 남은 것: J7-C 전력계(블루투스 연결 안 됨). Pi 64px·r101/r152는 완료, Jetson Nano는 제외, Hailo는 생략(11-20 사용자).

### 이유 분석 B~D 재실행: 기본 FedAPEN (11-20 18시 반, 사용자 지시)
- native head FedAPEN(`an_*`)은 R330에 공유 모델이 14%로 약해서, Dir 0.1에서 앙상블이 공유 + 비율보다 2.4 높았다(명제 1의 전제인 "공유 모델이 사후확률을 잘 추정"이 깨진 상태). 그래서 공유 모델이 강한 기본 FedAPEN(pool projector 512 + 512×C head, 본 표 FedAPEN)으로 같은 6개 데이터를 다시 돌린다: `anp_<데이터>_s0`. 사전 등록 기준은 위 표와 같다.
- `an_*`는 끝까지 두고 "공유 모델이 약할 때"의 대조로 쓴다(공유 모델 정확도별 차이 그림).

### 결과 (11-20 19시 반)
**이유 분석, native head FedAPEN `an_*` (R500 최종, seed 0)** — R330 중간값(앙상블 > 공유 + 비율)은 공유 모델이 덜 학습된 탓이었고, R500에서는 예측대로 뒤집혔다.

| 실험 | 데이터 | private | 공유 | 공유 + 비율 | 앙상블 | 앙상블 − (공유 + 비율) | 판정 |
|---|---|---|---|---|---|---|---|
| B | Dir 0.1 | 43.86 | 20.24 | **46.99** | 45.50 | −1.49 | 통과: 모든 빈도 구간에서 공유 + 비율 ≥ private (다수 +3.35, 중간 +1.97, 소수 +0.49) |
| C | Dir 0.3 | 31.84 | 29.60 | **46.19** | 41.32 | −4.87 | 통과 |
| C | Dir 1.0 | 20.18 | 34.19 | **40.94** | 35.69 | −5.25 | 통과. 비율 이득 +26.8(α0.1) → +16.6(0.3) → +6.8(1.0) |
| D | IID + feature shift | 10.34 | 16.58 | 15.85 | **19.09** | **+3.24** | 통과(> +1): feature shift에서는 private이 비율 이상을 더함 |
| D-대조 | IID | 11.83 | 33.49 | 32.29 | 33.52 | +1.23 | **기준(≤ 1) 근소 미달.** 단 공유 단독 대비 +0.03 → private 기여 0. 미달분은 IID에서 추정 비율의 잡음이 공유 + 비율을 깎은 것 |
| D-혼합 | Dir 0.1 + feature shift | 41.30 | 11.63 | 33.01 | **41.89** | **+8.88** | 통과 |

- 보정 강도 τ: 최적 τ = 1(Dir 0.1: 46.99, 0.5는 46.14, 1.5는 45.02), Dir 0.3도 1, Dir 1.0은 0.5(41.17)와 1(40.94)이 거의 같음 → 명제 1의 Bayes 보정(τ = 1) 예측과 일치.
- 기본 FedAPEN(`anp_*`)으로 같은 표를 다시 확인 중(자정 무렵).

**FedScout learner 기여(결합 − scout, seed 0)**: 라벨 이동만(CIFAR) 매 라운드 +0.6 / 2단계 +0.5 → IID + feature shift 매 라운드 **+1.13** / 2단계 +0.16 → Dir 0.1 + feature shift 매 라운드 **+3.15** / 2단계 **+1.96**. 사전 기준(상한에서 > +1) 두 설정 모두 통과(IID + shift는 근소). label shift 설명과 같은 방향.

**Flowers 동결 뒤 로컬 epoch (seed 3개)**
| 구성 | seed 평균 곡선 최고 | 마지막 평가 | learner 기여(마지막) | 학습 연산 (FedAPEN 대비) |
|---|---|---|---|---|
| 490R + 10 로컬 (lafsa8) | 69.32 ± 1.27 | 68.21 | −0.28 | 43.8 TFLOPs (6.6배 적음) |
| 490R + 50 로컬 (lafsa8L50) | 69.62 ± 0.99 | 69.62 | +0.91 | 49.8 TFLOPs (5.8배 적음) |
| 매 라운드 learner (npsa8) | 70.15 ± 1.25 | 69.97 | +1.28 | 257 TFLOPs (1.1배) |
- 50 로컬 epoch이면 npsa8과 차이 −0.53(최고), −0.35(마지막). 100 epoch은 seed 0만 끝났고 기여 +0.82로 50과 같음 → 50에서 포화로 보임(seed 3개 확인 중).

### feature shift seed 3개 (11-20 19시 40분, 사용자: "논문에 넣어야지")
- 2단계 FedScout `lafsa8L50_{Cifar100_iid_fs,Cifar100_fs}_s{0,1,2}`: `-gr 540`, R490 동결 뒤 로컬 50 epoch. R500 평가 = 로컬 10 epoch, R540 평가 = 로컬 50 epoch(한 run으로 둘 다).
- 상한 `npsa8_*fs_s{1,2}`, native head FedAPEN 분석 `an_{Cifar100_iid_fs,Cifar100_fs,Cifar100_iid}_s{1,2}` (seed 0은 완료).
- 판정 기준은 seed 0과 같다(사전 등록 그대로). seed 3개 평균 ± 표준편차로 보고한다.
- 라벨 이동 쪽 B·C도 seed 3개로(사용자 11-20): `an_{Cifar100,Cifar100_a0.3,Cifar100_a1}_s{1,2}`, feature shift 큐가 모두 뜬 뒤 시작.

### 기기 실측 2차 (11-20 20시): Pi 5 벽 전력 + 로컬 단계 E = 50 반영
- J7-C 인라인 USB 측정기(master, BLE)로 벽 전력을 기록하며 Pi 측정을 다시 했다(대기 약 5.0W, CPU 4코어 부하 8.4–8.8W로 측정기 대응 확인). 원자료 `docs/figs/data/device/wall_*.csv`, 측정기 로그 `wall_power.csv`(→ `wall_power_epoch.log`), 채움 `system/device_power_fill.py`. `device_compose.py`는 FedScout 로컬 단계를 50 epoch으로 계산(E_LOCAL).

| 기기·데이터 | FedScout 2단계(490R + 50 로컬): 평균 시간 / 가장 느림 / 평균 에너지 | FedAPEN | 배수(시간·에너지) |
|---|---|---|---|
| Pi 5, CIFAR (벽 전력) | 3.6 h / 6.8 h / 80 kJ | 27.7 h / 91.3 h / 888 kJ | 7.8배 · 11배 |
| Pi 5, Flowers (벽 전력) | 1.8 h / 2.5 h / 61 kJ | 6.0 h / 16.6 h / 200 kJ | 3.3배 · 3.3배 |
| Orin Nano, CIFAR | 0.25 h / 0.42 h / 6.7 kJ | 0.98 h / 2.62 h / 32.4 kJ | 3.9배 · 4.8배 |
| Orin Nano, Flowers | 0.05 h / 0.07 h / 1.8 kJ | 0.14 h / 0.38 h / 7.7 kJ | 3.0배 · 4.3배 |
- 가장 느린 클라이언트의 FedScout 시간 대부분은 통신 없는 로컬 단계라 다른 기기를 기다리게 하지 않는다. 연합 단계 라운드 시간은 모든 기기가 같은 c8이라 모델 이질성과 무관.
- 같은 run의 PMIC 값(Pi CIFAR): FedScout 45 kJ vs FedAPEN 624 kJ. 벽 전력이 대기 전력(약 5W)을 포함해 더 크다.


## 9. 외부 리뷰(GPT, 11-20 밤) 대응 — 사용자: "지피티가 하란 거 다해"
리뷰 판정: 지금 제출이면 Weak Reject, ICML 목표로 9주 더 쓸 만함. **12월 중순에 (a) 학습 중 지식 전달을 분리한 대조와 (b) 실제 도메인 결과를 보고 ICML 제출 여부를 판단**한다(리뷰 권고 그대로 채택).

| # | 리뷰 지적 | 대응 | 상태 |
|---|---|---|---|
| 1 | FedAPEN 공유 모델은 학습 중 private의 KL·앙상블 손실로 도움을 받았다 → 공유 + 비율 비교는 추론 결합만 본다 | `--apen_shared_ce_only`(공유 모델은 CE만, 나머지 동일) 추가. `apence_{Cifar100,TinyImagenet}_s{0,1,2}`의 공유 + 비율을 기본 FedAPEN(61.15/48.33)과 비교 | 코드·smoke 완료, CIFAR 3개 큐 맨 앞, Tiny 3개는 최종 큐 끝 |
| 2 | scout 단독을 본 표에 | `system/main_table.py` → `docs/figs/main_table.md`: scout 단독 / scout + 비율 / 결합 행, 마지막 평가 값, FedAPEN 대비 seed별 짝지은 차이 | 완료(CIFAR scout 단독 65.55, 결합 66.05) |
| 3 | 정확도 차이를 2단계에 귀속하지 말 것, 기준선에도 같은 작은 탐색을 | 2단계는 "같은 정확도에서 연산 절감"으로만 주장. 탐색: FedAPEN 공유 lr ×{1,3} × head ×{1,8}(`--apen_shared_head_lr_scale` 추가), FedScout lr ×{1,3} × head ×{1,8}, CIFAR seed 0, validation(held-out)으로 선택 → 고른 설정만 seed 3개 | 큐 맨 앞 6개 |
| 4 | test 곡선 최고값 대신 validation 선택 + 짝지은 차이 | 마지막 평가 값과 짝지은 차이를 표에 추가. FedAPEN에 held-out 정확도(`APENVal`) 기록 추가. FedScout은 held-out 결합 정확도(`held_counts`)로 선택 가능. 기준선(ariel)은 validation 기록이 없어 마지막 라운드 값으로 대체 | 부분 완료 |
| 5 | C2가 가장 약함, τ = 1은 온도 의존, 기준 비율·평활화 명시 | `theory_label_shift.md`에 구현 세부 추가, NLL 기록(`APENNLL`, `nll_*`), 보정 사다리 진단(`scout_bias`: 학습된 클래스별 bias) 추가 | 완료(새 run부터 기록) |
| 6 | 보정 사다리: prior → 클래스별 bias → 입력 의존 잔차, 로컬 표본 수·공유 모델 품질을 따로 바꾸며 정확도와 log loss | 공유 모델 품질 축은 궤적 그림(아래 7)으로. 로컬 표본 수 축은 `-fs`(few-shot, 클래스당 표본 수 제한) 또는 50/100명 분할로 → 실행 계획 필요 | 진단 코드 완료, 표본 수 축 실험 미착수 |
| 7 | R330 → R500 역전을 공유 모델 품질 대비로 | `system/apen_trajectory.py` → `docs/figs/apen_trajectory_native.png`: 라벨 이동은 공유 모델이 좋아질수록 앙상블 이득이 0 아래로(Dir 0.1: 공유 약 18%에서 0), feature shift는 +3 유지 | 완료(native head, seed 0) |
| 8 | D-대조 미달을 그대로, D-혼합은 "단독 최고 대비 +0.59" | 보고 방식 확정: D-대조 "사전 기준 미달(+1.23), 공유 단독 대비 +0.03", D-혼합 "공유 + 비율 대비 +8.88이지만 private 단독 대비 +0.59(공유 모델이 feature shift에서 11.6%로 무너짐)" | 문서 반영 |
| 9 | FedScout에서 feature shift일 때 learner 이득이 일관되게 크다고 쓰지 말 것 | E = 50 seed 3개(`lafsa8L50_*fs`) 결과 뒤 판단 | 진행 중 |
| 10 | 선행: Xu & Huang, FedRoD, PersFL, FedType + FedSCE(CVPR 2025), FedEMoE(ICML 2026) | `related_work.md`에 두 편 추가(존재 확인) | 완료 |
| 11 | 두 번째 이중 모델(FML 등)에서도 분해 재현 | FML에 같은 진단(공유 + 비율, 클래스별 기록, NLL) 추가 후 CIFAR Dir 0.1 / feature shift | 다음 작업 |
| 12 | 실제 도메인: DomainNet 공통 10클래스, 도메인 간 클래스 비율을 맞춘 조건 먼저, 모델 배정과 도메인 배정을 독립으로 | DomainNet 6도메인 다운로드 중(FedBN 10클래스만 추출) → 분할 생성기 작성 | 다운로드 진행 중 |
| 13 | ICML 본문은 8쪽 | 일정 표 수정 | 완료 |
| 14 | 실기기는 step 비용 × 횟수의 추정. 소규모 실제 연속 라운드(통신 포함)와 로컬 50 epoch을 재서 추정과 비교, 동기 라운드 지연·최고 메모리·cascade 정확도–지연 곡선 | Pi·Orin에서 짧은 실제 FL(통신 포함) 측정 스크립트 필요. cascade 곡선은 로그 + 기기 추론 지연으로 계산 가능 | 미착수(TMC 전 필수, ICML은 보조) |
| 15 | 같은 scout에 learner를 붙이는 추가 비용과 그 대가 | 표로 정리(로컬 단계 연산 vs learner 기여) | 다음 작업 |
