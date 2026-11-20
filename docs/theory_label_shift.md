# 이유 분석 A: label shift에서 개인화의 상한 (2026-11-20 초안)

논문의 "왜 private이 분포만 더하나"를 설명하는 이론 절 초안. 실험 B~D와 사전 등록 기준은 `docs/paper_story.md` 8절.
새 정리가 아니라 고전 결과(Saerens et al. 2002, Lipton et al. 2018, Menon et al. 2021)를 이기종 FL 개인화 문제에 옮긴 것이다. 기여는 "이 결과가 HtFL 벤치마크 설정에서 무엇을 뜻하는지"를 밝히고 실험으로 확인하는 데 있다.

## 설정과 기호

- 클라이언트 k의 데이터 분포: p_k(x, y) = p_k(x | y) π_k(y). K는 클래스 수, π_k는 지역 클래스 비율.
- 공유 모델은 모든 클라이언트 데이터를 합친 분포 p(x, y) = Σ_k w_k p_k(x, y)를 학습한다고 본다(w_k는 데이터 비율). 합친 클래스 비율은 π(y) = Σ_k w_k π_k(y).
- 공유 모델의 logit g(x)가 사후확률을 맞게 추정한다고 가정한다: softmax(g(x)) = p(y | x).

## 명제 1 (label shift: 공유 모델 + log 비율이 클라이언트별 Bayes 최적)

**가정** (label shift): 모든 k에 대해 p_k(x | y) = p(x | y). 즉 클래스 안의 데이터 모양은 같고 비율만 다르다. Dirichlet label skew 분할은 정의상 이 가정을 만족한다(클래스별 표본을 클라이언트에 나눠 줄 뿐이다).

**결론**: 클라이언트 k의 사후확률은
  p_k(y | x) = p(y | x) · [π_k(y) / π(y)] / Z_k(x),  Z_k(x) = Σ_y' p(y' | x) π_k(y') / π(y').
따라서 클라이언트 k의 Bayes 최적 분류기는
  ŷ_k(x) = argmax_y [ g_y(x) + log π_k(y) − log π(y) ].

**증명**: Bayes 정리로 p_k(y | x) = p(x | y) π_k(y) / p_k(x)이고, p(x | y) = p(y | x) p(x) / π(y)를 넣으면 p_k(y | x) = p(y | x) · π_k(y) / π(y) · p(x) / p_k(x). 마지막 인수는 y와 무관하므로 정규화 상수에 흡수된다. ∎

**따름정리 1 (충분성)**: label shift에서 공유 모델의 사후확률이 주어지면, 클라이언트 k에 필요한 개인화 정보는 K − 1개의 숫자 π_k뿐이다. 클라이언트 데이터로 학습한 어떤 개인화 모델 h_k(private 모델, 공유·private 앙상블 등)도 모집단 수준에서는 위 분류기보다 위험을 낮출 수 없다(위 분류기가 이미 Bayes 최적이므로).

**따름정리 2 (유한 표본에서 private이 이길 수 있는 경우)**: 실제로 private이 공유 + 비율을 이기려면 둘 중 하나가 필요하다.
1. 공유 모델의 사후확률 추정 오차가 크다(용량 부족, FedAvg의 client drift).
2. π_k 추정 오차가 크다.
private은 지역 표본 n_k개만 보고 p_k(y | x) 전체를 추정하므로, 추정 분산이 공유 모델(전체 N개)보다 크다. 그래서 n_k ≪ N인 보통의 FL 설정에서 private의 나머지 기여는 대부분 과적합이다.

**CIFAR-100 Dir 분할에서**: 전체 데이터를 클라이언트에 나누므로 π는 균등이고, log π(y)는 상수라 빠진다. 우리 실험의 "공유 + 비율"은 g(x) + log π̂_k(y)이고, π̂_k는 지역 학습 데이터의 클래스 빈도다(빈 클래스는 1로 보정, `clientapen.py`).

## 명제 2 (feature shift: private이 필요한 경우)

**가정**: p_k(x | y)가 클라이언트마다 다르다.

**결론**: p_k(y | x) ∝ p(y | x) · [π_k(y) / π(y)] · r_k(x, y),  r_k(x, y) = p_k(x | y) / p(x | y).

**의미**: r_k는 x에 따라 달라지는 함수라 클래스별 숫자 벡터로 표현할 수 없다. 이것을 추정하려면 클라이언트 데이터로 학습한 모델이 필요하고, 이것이 이기종 private 모델이 정당하게 기여할 수 있는 자리다. 증명은 명제 1과 같은 대입이다.

## FedAPEN 관찰과의 연결

- FedAPEN의 추론은 w·p(x) + (1 − w)·g(x)이고, w는 클라이언트당 숫자 하나다.
- 명제 1이 맞는 설정에서 private logit p(x)는 대략 "log π_k + 과적합 잡음"이다. 숫자 하나 w로는 비율 효과와 잡음을 분리할 수 없다. 그래서 비율만 정확히 더하는 공유 + 비율이 앙상블보다 높게 나온다(CIFAR 61 vs 54.83, Tiny 48 vs 41.84).
- 같은 논리로, 우리 FedScout의 learner 기여가 0에 가까운 것도 설명된다.

## 실험으로 확인할 예측 (사전 등록: `docs/paper_story.md` 8절)

| 예측 | 실험 |
|---|---|
| label shift에서 공유 + 비율 ≥ 앙상블, private의 우위는 지역 다수 클래스에만 | B (Dir 0.1, 클래스 빈도 3분위) |
| skew가 약해지면 비율 이득이 줄지만 순서는 유지 | C (Dir 0.3, 1.0) |
| feature shift에서는 앙상블 > 공유 + 비율 | D (IID + feature shift, 대조 IID, 혼합 Dir 0.1 + feature shift) |

**추가하면 좋은 싼 검증 (아직 안 함)**: 명제 1은 보정 계수 τ = 1을 예측한다(g + τ·log π̂_k). 공유 모델이 잘 보정돼 있으면 최적 τ가 1 근처여야 한다. 평가 때 τ ∈ {0, 0.5, 1, 1.5, 2}의 정확도를 같이 세면 된다. 최적 τ가 1보다 훨씬 크면 공유 모델이 과소 확신 상태라는 뜻이다.

## 한계 (논문에 명시)

- 공유 모델이 합친 분포의 사후확률을 맞게 추정한다는 가정은 근사다. 강한 skew에서 FedAvg는 client drift로 편향된다.
- 명제는 모집단 수준이다. 유한 표본 논증(따름정리 2)은 정성적이다. 정량적 경계는 없다.
- 실제 기기 데이터는 label shift와 feature shift가 섞여 있다. D-혼합 실험이 이 경우를 본다.

## 인용 (확인함)

- M. Saerens, P. Latinne, C. Decaestecker, "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure," Neural Computation 14(1):21–41, 2002. DOI 10.1162/089976602753284446
- Z. C. Lipton, Y.-X. Wang, A. Smola, "Detecting and Correcting for Label Shift with Black Box Predictors," ICML 2018, PMLR 80:3122–3130. https://arxiv.org/abs/1802.03916
- A. K. Menon et al., "Long-tail learning via logit adjustment," ICLR 2021. https://arxiv.org/abs/2007.07314 (이번에 다시 확인하지 않음)
