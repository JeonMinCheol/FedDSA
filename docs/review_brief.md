# 외부 검토용 정리: 이기종 연합학습(HtFL)의 지식 공유 재검토 (2026-11-20 기준)

> 이 문서는 다른 AI/연구자에게 그대로 붙여 넣어 비판적 검토를 받기 위한 자기완결형 정리다. 좋은 점보다 **약점·겹치는 선행 연구·위험**을 찾아 달라는 것이 목적이다. 숫자는 모두 실제 로그에서 뽑았고, seed 수와 미완료 항목을 표시했다.

## 0. 검토자에게 묻고 싶은 것 (맨 끝 9절에 구체적 질문)
1. 이 기여 조합(진단 + 이론 + 2단계 설계 + 실기기 효율)이 **ICML 2027**(본문 마감 2027-01-22)에 낼 만한가? 안 되면 IEEE TMC로 간다.
2. 리뷰어가 가장 먼저 공격할 지점은 어디이고, 9주 안에 막을 수 있는가?
3. 겹치는 선행 연구를 우리가 놓쳤는가?

---

## 1. 문제 설정
- **모델 이기종 연합학습(HtFL)**: 클라이언트(IoT 기기)마다 제조사·하드웨어가 달라 서로 다른 구조의 모델(private model)을 쓴다. 서버는 이 모델들을 직접 평균할 수 없다.
- 기존 HtFL은 구조가 달라도 통하는 "통로"로 지식을 나눈다: prototype(FedProto, FedTGP), anchor(FedSA), head/logit(LG-FedAvg, FedGH), 생성기/그래프(FedRG), 그리고 **작은 공유 모델 + private 상호 학습/앙상블**(FML, FedKD, FedAPEN, FedMRL, pFedMoE).
- 우리 설정(사용자 지정, IoT 가정): 데이터 증강 없음, Dirichlet label skew α = 0.1(극단적), 클라이언트 20명, 모델 구성 HtFE-img-8(클라이언트 i는 FedAvgCNN, GoogLeNet, MobileNetV2, ResNet18/34/50/101/152 중 i % 8), 500 라운드, 로컬 1 epoch, seed 3개, **평가 지표 = seed 평균 정확도 곡선의 최고값**(10라운드마다 개인화 test 정확도). 데이터: CIFAR-100, Flowers102, Tiny-ImageNet.

## 2. 핵심 주장 (기여 4개)
- **C1 진단**: label skew 벤치마크에서 이기종 private 모델은 "지역 클래스 비율" 이상을 거의 더하지 못한다. 학습 없는 기준 **"공유 모델 logit + log(지역 클래스 빈도)"**(이하 *공유 + 비율*)가 FedAPEN의 학습된 앙상블과 같거나 낫다. prototype/anchor 계열 HtFL 기준선은 Local과 큰 차이가 없다(관찰 1). 즉 **지식은 같은 구조의 가중치 평균으로만 의미 있게 흐르고, 개인화의 나머지는 비율 보정**이다.
- **C2 이유(이론)**: label shift(p_k(x|y) = p(x|y), p_k(y)만 다름)에서는 클라이언트 k의 Bayes 최적 분류기가 `argmax_y [g_y(x) + log π_k(y) − log π(y)]`(g = 합친 분포의 사후 logit)다(명제 1, 고전 결과: Saerens et al. 2002, Lipton et al. 2018, Menon et al. 2021). 따라서 공유 모델이 사후확률을 잘 추정하면 **어떤 통로로도 private이 공유 + 비율을 넘을 수 없다**(모집단 수준). feature shift(p_k(x|y)가 다름)에서는 x에 따라 달라지는 보정 항 r_k(x, y) = p_k(x|y)/p(x|y)가 생겨 private이 정당하게 기여할 수 있다(명제 2). 유한 표본에서는 private이 n_k개만 보므로 나머지 기여는 대부분 과적합이다.
- **C3 설계(FedScout, 2단계)**: 진단·이론에 맞춰 연합은 모든 기기가 같은 구조로 갖는 **작은 c8 scout**(CIFAR ResNet-8, MLPerf Tiny 크기, 8.5만 파라미터)만 하고, 이기종 private 모델(learner)은 **연합이 끝난 뒤 로컬에서만** 학습한다(scout 위의 잔차, 확신 없는 샘플만, cascade 추론).
- **C4 효율**: 같은 정확도대에서 기기 학습 연산·시간·에너지가 FedAPEN 등 이중 모델 방법보다 수 배 적다. Raspberry Pi 5와 Jetson Orin Nano에서 실측했다.

## 3. 방법: FedScout 최종 구성 (2026-11-20 확정)
- **scout**: c8, projector 없음(자기 분류기 64×C). 모든 클라이언트가 같은 구조. **모델 전체(head 포함)를 FedAvg**. scout lr 0.03(= private lr 0.01 × 3), **head lr ×8**(head만 lr 0.24).
- **1단계(R1–R490, 연합)**: 각 기기는 scout만 CE로 학습. learner는 학습하지 않음(`--scout_learner_after_freeze`). 라운드당 통신 = scout 전체 84.6K floats(CIFAR), 업·다운 각각.
- **R490**: scout 동결 + 지역 데이터로 head 개인화.
- **2단계(로컬 50 epoch, 통신 없음)**: 이기종 learner를 scout logit 위 잔차로 학습: `out = learner(x) + scout_logit(x)`(scout 고정). **screening**: 고정된 scout가 정답 클래스 확률 ≥ 0.5로 맞히는 샘플은 건너뜀(로그상 learner는 샘플의 약 30%만 학습).
- **추론**: `scout + α·learner`, α ∈ {0, 0.1, 0.25, 0.5, 0.75, 1}를 학습 데이터 held-out 5%로 선택. cascade(scout가 확신하면 learner를 돌리지 않음)로 learner는 샘플의 30~35%에서만 실행.
- 표기: 동결 뒤 50 epoch은 통신이 없으므로 **"490R 연합 + 50 로컬 epoch"**으로 기준선의 500R와 따로 표시한다. 평가 때 head를 지역 데이터로 1 epoch 미세 조정(`--scout_eval_ft`, 개인화 평가)한다.
- 설계 변천(투명성): 처음엔 쌍둥이 잔차 + 개인 head + pool projector + 매 라운드 learner였다. 실험으로 ① head까지 평균(shareall)이 개인 head보다 좋고(Flowers +4.45), ② projector 제거로 통신이 FedAPEN native head와 같아지고, ③ head lr ×8이 큰 이득(Flowers 64.53 → 70.15), ④ learner를 연합 중 학습하지 않아도 정확도가 거의 같다는 것을 확인해 지금 구성이 됐다. **lr ×8은 우리 하이퍼파라미터 탐색 결과이고, 기준선은 원 논문 lr(0.01)이다.**

## 4. 결과

### 4.1 본 표 (seed 평균 곡선 최고 ± 표준편차, seed 3개)
| 방법 | CIFAR-100 | Flowers102 | Tiny-ImageNet |
|---|---|---|---|
| FedProto | 38.24 ± 0.09 | 49.88 ± 1.11 | 18.24 ± 0.22 |
| FedGH | 39.83 ± 0.26 | 53.06 ± 0.66 | 23.19 ± 0.13 |
| LG-FedAvg | 40.64 ± 0.41 | 53.80 ± 0.51 | 23.64 ± 0.14 |
| FedRG | 41.98 ± 0.11 | 50.62 ± 0.39 | 24.15 ± 0.15 |
| FedTGP | 44.51 ± 0.27 | 55.60 ± 0.53 | 23.20 ± 0.37 |
| FedSA | 44.89 ± 0.29 | 55.59 ± 0.19 | 23.92 ± 0.05 |
| FedMRL (공유 c8) | 38.24 ± 0.11 | 48.79 ± 0.68 | 진행 중 |
| pFedMoE (공유 c8) | 40.77 ± 0.43 | 진행 중 | 진행 중 |
| FedAPEN (공유 c8, 512×C head) | 54.83 ± 0.27 | 59.74 ± 0.44 | 41.84 ± 0.32 |
| **FedScout, 매 라운드 learner(npsa8, 500R)** | 66.23 ± 0.78 | 70.15 ± 1.25 | — |
| **FedScout 2단계, 490R + 10 로컬** | 66.05 ± 0.91 | 69.32 ± 1.27 | — |
| **FedScout 2단계, 490R + 50 로컬 (최종)** | 실행 대기(11-21) | **69.62 ± 0.99** | 실행 대기 |

- 공유 모델 방법은 모두 c8 공유 모델로 맞췄다(원 논문은 더 큰 공유 모델을 쓰기도 함; c20으로 돌린 FedAPEN 결과도 따로 있음). FedAPEN은 500R에서 아직 오르는 중이라(최고점 = R500) 1000R run을 ariel에서 돌리는 중.
- **scout 단독**(= c8 FedAvg + head 미세 조정, private 없음): CIFAR 2단계 R500에서 seed별 65.96 / 64.13 / 66.56(결합값 66.22 / 64.86 / 67.06). 즉 **최종 정확도의 대부분은 scout 단독이 낸다.**
- 아직 없음: Tiny 최종 구성, CIFAR 최종(E=50), Local 기준선 표 값, FedAPEN native head(통신 맞춘 버전) 본 표 값, 클라이언트 수 50/100(ariel, 11-24 예정).

### 4.2 진단: FedAPEN 분해 (C1)
**기본 FedAPEN(본 표 버전), R500, seed 0/1/2**
| 데이터 | private 단독 | 공유 단독 | **공유 + 비율** | 학습된 앙상블(본 표 값) |
|---|---|---|---|---|
| CIFAR-100 | 43.90 / 44.31 / 44.09 | 32.81 / 33.40 / 34.22 | **60.69 / 61.15 / 61.57** | 54.83(평균) |
| Tiny-ImageNet | 28.95 / 28.70 / 29.04 | 23.63 / 23.59 / 23.28 | **48.71 / 48.38 / 47.91** | 41.84(평균) |
- 학습 없는 공유 + 비율이 FedAPEN 앙상블보다 CIFAR +6.3, Tiny +6.5 높다. 앙상블 가중치(w_private 평균 0.6)는 비율 효과와 private의 과적합을 숫자 하나로 섞어 둘을 분리하지 못한다.

### 4.3 이유 검증: label shift vs feature shift (C2), native head FedAPEN, **seed 0, R500** (seed 1·2 진행 중)
| 설정 | private | 공유 | 공유 + 비율 | 앙상블 | 앙상블 − (공유 + 비율) | 사전 등록 기준 |
|---|---|---|---|---|---|---|
| B: Dir 0.1 | 43.86 | 20.24 | **46.99** | 45.50 | −1.49 | 통과(모든 지역 빈도 3분위에서 공유 + 비율 ≥ private: 다수 +3.35, 중간 +1.97, 소수 +0.49) |
| C: Dir 0.3 | 31.84 | 29.60 | **46.19** | 41.32 | −4.87 | 통과 |
| C: Dir 1.0 | 20.18 | 34.19 | **40.94** | 35.69 | −5.25 | 통과(비율 이득 +26.8 → +16.6 → +6.8로 α에 따라 감소) |
| D: IID + feature shift | 10.34 | 16.58 | 15.85 | **19.09** | **+3.24** | 통과(> +1) |
| D-혼합: Dir 0.1 + feature shift | 41.30 | 11.63 | 33.01 | **41.89** | **+8.88** | 통과 |
| D-대조: IID, shift 없음 | 11.83 | 33.49 | 32.29 | 33.52 | +1.23 | **근소 미달(≤ 1 기준)**. 공유 단독 대비 +0.03 → private 기여 0. IID에서는 추정 비율의 잡음이 공유 + 비율을 깎음 |
- **보정 강도 τ**(g + τ·log π̂): 최적 τ가 Dir 0.1에서 1(46.99; τ 0.5 46.14, 1.5 45.02), Dir 0.3에서 1, Dir 1.0에서 0.5와 1이 거의 같음 → Bayes 보정(τ = 1) 예측과 일치.
- feature shift는 **합성**이다: 같은 CIFAR-100에서 클라이언트 i가 도메인 i % 5(원본, 흑백, 색 반전, 90도 회전, 16×16 픽셀화)로 이미지를 본다. 라벨 분할과 무관.
- 주의: native head FedAPEN은 R330에서 공유 모델이 14%로 약해 **앙상블 > 공유 + 비율(+2.4)**이었다가 R500에서 뒤집혔다. 즉 진단은 "공유 모델이 충분히 학습됐을 때"라는 전제(명제 1의 가정)에 의존한다. 기본 FedAPEN으로도 같은 6개 설정을 다시 돌리는 중.

### 4.4 FedScout에서 learner 기여 = 결합 − scout 단독 (seed 0; feature shift seed 3개 진행 중)
| 설정 | 매 라운드 learner | 2단계(10 로컬) |
|---|---|---|
| 라벨 이동만(CIFAR, seed 3개 평균) | +0.61 | +0.50 |
| 라벨 이동만(Flowers, seed 3개 평균) | +1.28 | −0.28 (50 로컬: +0.91) |
| IID + feature shift | +1.11 | +0.16 |
| Dir 0.1 + feature shift | +3.05 | +1.96 |
- 이론 방향과 같다: 라벨 이동만이면 learner 기여 ≈ 0.5~1, feature shift가 끼면 커진다.

### 4.5 2단계 vs 매 라운드, 로컬 epoch 수 (Flowers, seed 3개)
| 구성 | 최고 | learner 기여 | 학습 연산/클라이언트 (FedAPEN 290 TFLOPs 대비) |
|---|---|---|---|
| 490R + 10 로컬 | 69.32 ± 1.27 | −0.28 | 43.8 (6.6배 적음) |
| **490R + 50 로컬** | **69.62 ± 0.99** | +0.91 | **49.8 (5.8배 적음)** |
| 490R + 100 로컬 | 69.52 ± 0.94 | +0.81 | 57.2 (5.1배 적음) |
| 매 라운드 learner | 70.15 ± 1.25 | +1.28 | 257 (1.1배) |
- CIFAR(E=10): 66.05 vs 매 라운드 66.23, 연산 80.9 vs FedAPEN 544 TFLOPs(6.7배 적음). FLOPs는 `torch.utils.flop_counter` 실측(depthwise conv의 groups 버그 보정; PyTorch 이슈 #205650으로 보고함).

### 4.6 실기기 측정 (C4)
- Raspberry Pi 5(CPU 4코어, 4GB; 전력 = 인라인 USB 측정기 J7-C 벽 전력, 대기 약 5.0W), Jetson Orin Nano Super(GPU; 전력 = tegrastats VDD_IN 보드 입력). 각 구성 요소의 한 SGD step(배치 32)을 기기에서 재고, 실제 분할의 클라이언트별 데이터 수와 라운드 수로 전체 학습을 합성했다(전체 FL을 기기에서 끝까지 돌린 것은 아님). FedScout = 490 × scout 학습 + 50 × 로컬 learner(screening 30%).

| 기기·데이터 | 지표 | FedScout 2단계 | FedAPEN | private만 매 라운드(LG/Proto/TGP/SA) |
|---|---|---|---|---|
| Pi 5, CIFAR | 클라이언트 평균 학습 시간 | **3.6 h** | 27.7 h (7.8배) | 25.4 h |
| | 가장 느린 클라이언트(r152) | 6.8 h* | 91.3 h | 89.1 h |
| | 평균 에너지(벽 전력) | **80 kJ** | 888 kJ (11배) | 804 kJ |
| Pi 5, Flowers | 평균 시간 / 에너지 | 1.8 h / 61 kJ | 6.0 h / 200 kJ (3.3배) | 4.6 h / 143 kJ |
| Orin Nano, CIFAR | 평균 / 가장 느림 / 에너지 | 0.25 h / 0.42 h / 6.7 kJ | 0.98 h / 2.62 h / 32.4 kJ (3.9배) | 0.82 h / 2.46 h / 27.4 kJ |
| Orin Nano, Flowers | 평균 / 에너지 | 0.05 h / 1.8 kJ | 0.14 h / 7.7 kJ (3.0배) | 0.11 h / 5.9 kJ |
- *연합 단계(490R)는 모든 기기가 같은 c8만 학습하므로 **라운드 시간이 모델 이질성과 무관**하다(낙오자 없음). 가장 느린 클라이언트의 6.8 h 대부분은 통신 없는 로컬 단계라 다른 기기를 기다리게 하지 않는다. FedAPEN은 매 라운드 ResNet-152 기기가 라운드 시간을 정한다.
- Orin(GPU)에서는 배치가 작아 screening이 learner step 시간을 거의 줄이지 못한다; 절감은 연합 중 learner를 학습하지 않는 데서 나온다. Pi(CPU)에서는 screening이 step 시간을 약 절반으로 줄인다.
- Jetson Nano(4GB)는 다른 사용자 프로세스 때문에 r34부터 스왑에 빠져 제외했고, Hailo NPU는 컴파일러가 없어 생략했다.

## 5. 선행 연구와 겹치는 점 (전체 목록: `docs/related_work.md`)
| 우리 요소 | 겹치는 선행 연구 | 새로움 주장의 범위 |
|---|---|---|
| 사후 log-prior 보정 자체 | Saerens 2002, Lipton 2018(BBSE), Menon 2021(logit adjustment), Balanced Softmax 2020 | **새롭지 않음.** 우리는 이를 HtFL 진단 기준으로 쓴다 |
| FL에서 prior/분포 보정 | FedLC(ICML 2022), FedRS(KDD 2021), ReBaFL, FedRoD(ICLR 2022: personal head ≈ 분포 함수), FedTHE(ICLR 2023), Xu & Huang 2023(균형 전역 + prior 보정 개인화) | 같은 구조 FL에서 이미 있음. **HtFL에서 "공유 + 비율 ≥ 이중 모델 앙상블"을 보이고 공유/private/비율로 분해한 연구는 찾지 못함** |
| 작은 공유 모델 + 이기종 private | FML, FedKD, ProxyFL, FedType, FedMRL, pFedMoE, FedAPEN | 모두 private을 **매 라운드** 학습. 연합 중 private 미학습 + 로컬 단계만은 찾지 못함 |
| 연합 뒤 로컬 적응 1회 | PersFL(2021: FedAvg 뒤 로컬 KD 1회, 같은 구조), Yu et al. 2020(FL 뒤 로컬 FT), FedAvg-FT, FedBABU, FedRep | **2단계 골격은 겹침(강한 부분 겹침).** 차이는 이기종 learner, 잔차 + screening + cascade, 그리고 진단이 설계를 정당화한다는 점 |
| 기기는 작은 모델만 학습 | FedGKT(큰 모델은 서버), HeteroFL/FjORD/FedRolex(sub-model) | 큰 모델이 서버에 있거나 별도 private이 없음 |
| 확신도 cascade / 샘플 선택 | SplitGP, FedHLM, Dirichlet-UQ PFL, FedBalancer, FedFT-EDS | 부분 겹침 |
| 구조 간 지식 공유의 한계 | FedTopo(2026: 구조가 다르면 공통 좌표가 유지되지 않음), HtFL survey | 문제의식 겹침(본문 미확인) |
- 본문을 읽은 것: PersFL, Yu 2020, ProxyFL, FedType, FedRoD, pFedMoE, FedMRL, FedLC, ReBaFL. 나머지는 초록·검색 요약 수준.

## 6. 위험과 약점 (우리가 아는 것, 솔직하게)
1. **"그냥 c8 FedAvg + fine-tuning 아니냐"**: 최종 정확도의 대부분이 scout 단독에서 나오고 learner 기여는 라벨 이동에서 +0.5~0.9. 방법의 노벨티가 얇아 보일 수 있다. 우리 답: 그것이 C1·C2의 귀결이며, 이기종 기기가 자기 모델을 유지한다는 전제에서 그 비용을 최소화하는 것이 설계의 몫. 설득력은 미지수.
2. **하이퍼파라미터 비대칭**: 우리는 scout lr ×3, head lr ×8을 탐색했고 기준선은 원 논문 lr. head lr ×1 ablation(진행 예정)과 "FedAPEN + 우리 발견" 부록으로 답하려 하지만, 기준선도 같은 예산으로 튜닝하라는 요구가 나올 수 있다.
3. **이론이 고전 결과의 이식**: 명제 1·2는 Saerens/Lipton/Menon의 Bayes 보정을 HtFL에 옮긴 것. 유한 표본·FedAvg 편향에 대한 정량적 경계가 없다. "언제 private이 이기나"를 정량화할 계획(공유 모델 품질, n_k의 함수).
4. **feature shift가 합성**: 회전·반전 등 인위적 변환. 실제 도메인 차이(DomainNet, Office-Caltech, Digit5)에서 확인해야 한다(계획).
5. **진단 범위**: 분해 분석은 FedAPEN 하나. FML/FedMRL/pFedMoE로 확대 계획.
6. **seed**: 4.3·4.4는 seed 0(seed 3개 진행 중). D-대조는 사전 기준을 근소하게 못 넘었다(공유 단독 대비 0이지만 기준은 공유 + 비율 대비로 정했음).
7. **전제 의존**: native head FedAPEN의 R330 → R500 뒤집힘처럼, 공유 모델이 덜 학습되면 진단이 반대로 나온다.
8. **평가 프로토콜**: "seed 평균 곡선의 최고값"은 test 곡선의 최댓값이라 모든 방법에 같은 낙관 편향이 있다. FedAPEN 등은 500R에서 수렴 전(1000R run 진행 중).
9. **2단계 정확도 손실**: Flowers에서 매 라운드 learner보다 −0.53(E=50). 사전 기준(0.5 이내)을 근소하게 넘었다(E=10 때 −0.83은 사용자가 연산 이득을 이유로 수용).
10. **540R vs 500R**: 로컬 50 epoch을 라운드로 세지 않는 것이 공정한지(통신 0이지만 연산은 포함해 보고).
11. **실기기는 합성**: 한 step을 재고 곱했다. 메모리 스왑, 열 제한, 통신 시간, 실제 FL 런타임은 반영 안 됨. Pi 4GB에서 큰 ResNet 학습 자체가 현실적인지도 질문거리.
12. **범위**: 이미지 분류, label skew 중심, 클라이언트 20명(50/100명은 ariel 진행 중), 지역 데이터가 적은 IoT 가정. 지역 데이터가 많은 cross-silo에서는 결론이 다를 수 있다.
13. **분야를 부정하는 인상**: "이기종 FL이 필요 없다"로 읽히면 HtFL 리뷰어가 방어적일 수 있다. 우리는 "label shift에서는 작은 공유 모델 + 비율이 상한이고, 이기종 모델의 가치는 feature shift와 배포에 있다"로 범위를 긋는다.

## 7. 논문 작성 계획
**제목(확정)**: *Rethinking Knowledge Sharing in Model-Heterogeneous Federated Learning: Share a Tiny Common Model, Specialize Locally* (방법 이름 FedScout는 본문에서 소개)

**투고**: ICML 2027(초록 2027-01-14, 본문 01-22) → 떨어지면 IEEE TMC. 저자는 1저자 단독 + 교신저자(지도교수). 1저자는 2027-08 졸업 예정, 학위논문은 이 연구로 작성(게재 여부와 무관).

**ICML 원고 구성(분석 논문 형식, 본문 9쪽)**
1. 서론: HtFL의 통로들이 무엇을 전달하는가라는 질문. 기여 C1–C4.
2. 설정·관련 연구.
3. 진단: HtFL 기준선 vs Local(관찰 1), FedAPEN 분해(공유 + 비율 ≥ 앙상블), 확대 진단(FML/FedMRL/pFedMoE, 계획).
4. 이유: 명제 1·2, 예측(τ = 1, α 의존, feature shift에서 역전), 실험 B·C·D(seed 3개), "언제 private이 이기나"(공유 모델 품질·n_k).
5. 설계: 2단계 FedScout(진단의 귀결로서).
6. 결과: 본 표, 통신·연산·라운드 대비 곡선, 실기기 표, ablation(평균 없음/개인 head/head lr ×1/screening 끔), 일반화(HtFE 2/3/4/5/9, lowend), 클라이언트 수.
7. 한계·범위.
- **TMC 버전**이면 실기기·시스템 비중을 키우고(에너지, 낙오자, cascade 지연), 이론은 축약.

**9주 일정**: 11/21–27 본 결과 완성·실제 feature shift 데이터 준비 → 11/28–12/04 진단 확대 → 12/05–12/18 "언제 private이 이기나" 정량화·그림 확정 → 12/19–01/08 원고·모의 심사 → 01/14 초록, 01/22 제출.

## 8. 진행 중·미완료
- 63 노드(RTX 2080 Ti × 5): feature shift seed 3개(FedScout 2단계·상한, FedAPEN 분석), B·C seed 3개, 기본 FedAPEN 분석(seed 0), 그 뒤 최종 구성 36개(CIFAR 본 표, ablation 12, 일반화 18, Tiny 3).
- ariel(다른 계정 2개): 기준선 마무리(pFedMoE, FedMRL, FedAPEN native head, FedAPEN 1000R, 클라이언트 수), 11-24 완료 예정.

## 9. 검토자에게 구체적 질문
1. C1–C4 중 ICML 리뷰어에게 가장 약하게 보일 기여는? 이론(C2)이 "고전 결과의 재포장"으로 보일 위험을 줄이려면 무엇을 보태야 하나(예: 유한 표본 경계, private이 이기는 조건의 정량화)?
2. "공유 + 비율 ≥ 이중 모델 앙상블"을 HtFL에서 보인 선행 연구를 알고 있나? 2단계(연합 중 private 미학습 + 로컬 단계) 설계와 같은 것은?
3. 4.3의 사전 등록 결과(특히 D-대조 근소 미달, native head의 R330→R500 역전)를 어떻게 보고해야 공정한가?
4. 실제 feature shift 데이터로 무엇이 가장 설득력 있나(DomainNet 10클래스 / Office-Caltech10 / Digit5 / 기타)?
5. 하이퍼파라미터 비대칭(우리만 head lr ×8) 문제를 9주 안에 어떻게 막는 게 가장 싸고 확실한가?
6. 실기기 측정을 "한 step × 횟수"로 합성한 방식이 TMC/ICML에서 받아들여질까? 최소한으로 무엇을 더 해야 하나?
7. 이 결과로 ICML이 현실적인가, 아니면 처음부터 TMC(또는 다른 곳)에 집중하는 게 나은가?
