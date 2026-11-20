# 논문 스토리와 실험 구성 (2026-11-19 밤 기준)

목표: **IEEE TMC**(11-20 결정, 안 되면 IoT-J). TMC는 실기기 평가와 이유 분석을 IoT-J보다 더 요구한다. 실험 규칙·프로토콜은 `docs/experiment_plan.md`, 진행 기록은 `system/logs/night_actions_20261117.md`, 기준선 측정은 `docs/baselines_progress.md`.

## 0. 제목 (11-20 사용자 결정)

**Rethinking Knowledge Sharing in Model-Heterogeneous Federated Learning: Share a Tiny Common Model, Specialize Locally**

- 방법 이름 FedScout는 제목에 넣지 않고 초록·본문에서 소개한다("scout"는 처음 보는 독자에게 뜻이 안 통함). IoT는 초록 첫 문장과 키워드로 드러낸다.
- "Specialize Locally"는 2단계 설계(연합 뒤 로컬 학습)가 통과하면 더 정확해진다.

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
| **기기 실측** | Raspberry Pi(CPU 학습), Jetson Nano(GPU 학습, tegrastats 전력), Hailo NPU(추론만). 학습: scout 한 step(연합 단계), 동결 뒤 learner 한 step(screening 30%/100%), 매 라운드 learner(npsa8), Local, FedAPEN. 추론: scout, private, cascade(= scout + 미확신 비율 × private). 전체 기기 시간 = epoch 수 × (n/B) × step 시간으로 조합 | **사용자가 직접 측정.** `system/device_bench.py`를 11-20 최종 구성(c8 native scout, projector 없음)으로 갱신, `--size 32/64 --classes 100/102/200`, Hailo용 `--export_onnx`(최신 torch는 `pip install onnx` 필요, 여기서는 export 미검증) |
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
