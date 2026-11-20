# ICML 2027 원고 구성 (11-20 밤, 외부 리뷰와 합의한 본문 골격)

**흐름**: ① private 모델이 학습과 추론에 각각 얼마나 기여하는가 → ② 어떤 조건에서 그 기여가 커지는가 → ③ 그 결과를 이용해 얼마나 싸게 학습할 수 있는가. 본문 결과는 **표 2개 + 그림 1개**. ICML 2027 `General` 유형(실증적 발견)으로 내며 새 정리는 붙이지 않는다. Bayes 보정 설명은 그림 1 앞에 **반 쪽**(수식 하나로 prior가 설명하는 부분을 짚고 바로 실험으로).

**생성**: `cd system && python paper_tables.py` → `docs/figs/paper_table1.md`, `paper_table2.md`, `paper_fig1.png` (끝난 run만 채우고 나머지는 running). 주 지표는 마지막 평가.

## 표 1. private 모델의 학습 기여와 추론 기여 (논문의 첫 결과, 원고는 이 표부터 쓴다)
| 열 | 출처 run |
|---|---|
| CE-only 공유 + prior | FedAPEN `apence_{Cifar100,TinyImagenet}_s{0,1,2}`(`--apen_shared_ce_only`), FML `fmlce_Cifar100_s*` |
| 결합 학습 공유 단독 / 공유 + prior | FedAPEN: ariel 기본 run R500(seed 3개; 튜닝 후 `apenhp` 선택 설정 × 3 seed로 교체), FML `fml_Cifar100_s*` |
| 방법 자체의 예측 | FedAPEN 학습된 앙상블, FML은 원래 출력이 private 모델(앙상블을 따로 맞추면 원래 출력과 구분해 표기) |
- 본문에서 해석하는 차이는 둘뿐: **(결합 공유 + prior) − (CE 공유 + prior) = 학습 중 private을 이용한 효과**, **(앙상블) − (결합 공유 + prior) = 같은 체크포인트에서 예측 규칙을 바꾼 효과**(현재 CIFAR −6.31, Tiny −6.49).
- 입력별 전달까지 주장하려면 클래스 평균 teacher 대조(`apentcm_Cifar100_s*`)를 함께 보고.
- CE 대조 결과에 따른 서론 주장(리뷰 2차):
  - 비슷하면: "검증한 label-skew 조건에서는 소형 공유 모델의 CE 학습과 로컬 prior 보정으로 결합 학습의 성능을 재현했다"(평균·seed별 차이, '실질적 동등'은 사전에 정한 허용 격차를 불확실성까지 만족할 때만).
  - 낮으면: "private 모델을 이용한 학습은 공유 모델을 개선했지만, 추론에서는 학습된 앙상블보다 공유 모델의 prior 보정이 더 정확했다"(학습과 추론에서 private의 가치가 다르다는 발견이 중심).

## 그림 1. private 모델의 이득이 남는 조건
- **왼쪽**: private 체크포인트(R500)를 고정하고 공유 체크포인트만 R50–R500으로 바꿔 끼운 대조(`apenswap_*`, `answap_*`; w·τ는 같은 held-out에서 선택). 세로축 = 앙상블 − (공유 + prior), 가로축 = 공유 모델의 검증 NLL. label shift(`Cifar100`)와 합성 feature shift(`Cifar100_iid_fs`)를 같은 패널에. 공동 학습 궤적(R40 +8.6 → R500 −1.5, `docs/figs/apen_trajectory_native.png`)은 교차 대조가 완성되면 부록으로. 15–18% 교차점은 현재 설정의 관측치로만.
- **오른쪽**: 개인화한 scout(head FT 포함) 위에 learner를 붙인 이득 = FedScout − scout 단독, 490R + 50 로컬. 조건: label shift만(CIFAR, Flowers) / label + feature shift(CIFAR) / feature shift(CIFAR IID) / **DomainNet**(이득이 작아도 본문에 남긴다: 실제 도메인에서 learner를 붙일 근거를 결정).

## 표 2. FedScout의 정확도와 학습 비용
- 행: 튜닝한 FedAPEN / 같은 체크포인트의 FedAPEN + prior / scout 단독 / 최종 FedScout. 열: 데이터셋별 마지막 평가 정확도 + 대표 데이터셋(CIFAR)의 클라이언트당 학습 연산.
- 꼭 보여줄 수치: Flowers **scout 68.71 → FedScout 69.62**, learner 추가 비용 **7.0 TFLOPs/클라이언트**(CIFAR는 50 로컬 run이 끝나면). 기기 시간·에너지는 대표 한 줄("기기에서 잰 step 비용으로 계산한 추정치", 실제 LAN FL 측정이 끝나면 함께).

## 부록
전체 HtFL 기준선 표, seed별 원값, 64×C / 512×C head 비교, 학습률 탐색 전 조합, α·라운드별 곡선, 10/50/100 로컬 epoch, 기기별 cascade 표, 공동 학습 궤적, 실제 LAN FL 측정 원자료. **단, 튜닝 후 순위가 바뀌거나 DomainNet에서 이득이 사라지는 결과는 결론을 바꾸므로 본문에 남긴다.**

## 8쪽 배분
서론·관련 연구 1.5쪽 / 설정 1쪽 / 표 1과 그림 1의 분석 3쪽 / FedScout 설계 1쪽 / 표 2와 한계 1.5쪽.
