# 관련 연구 정리 (2026-11-20, 63 노드)

논문: **Rethinking Knowledge Sharing in Model-Heterogeneous Federated Learning: Share a Tiny Common Model, Specialize Locally** (방법 이름 FedScout). 스토리와 기여는 `docs/paper_story.md`.

- 출처: 11-20 선행 연구 검색 3갈래(2단계 설계, 관찰 2(클래스 분포), 기기 연산 절감)와 제목 중복 검색, 11-19~20 기준선 lr 확인, 이전 세션의 FedRes 검토.
- **읽은 정도**: [본문] 본문을 읽음, [초록] 초록만, [요약] 검색 결과 요약만, [지식] 검색으로 확인하지 않고 알고 있던 것. 논문에 쓰기 전에 [요약]·[지식]은 본문으로 확인한다.
- **판정**(우리 아이디어 대비): 같음 / 부분 겹침 / 다름. "같음"은 하나도 찾지 못했다.
- URL이 비어 있는 항목은 이번에 확인하지 못한 것이다(추측으로 채우지 않음).

## 1. 우리 주장별 위치와 필수 인용

| 우리 주장 | 새로운 부분 | 새롭지 않은 부분 (인용 필수) |
|---|---|---|
| 관찰 1: 지식은 같은 구조끼리만 흐른다 | 증강 없는 강한 label skew에서 구조 간 채널이 Local과 비슷하다는 체계적 비교 | 기존 지식 공유 형태 분류와 한계 지적(FedTopo), HtFL survey |
| 관찰 2: 이기종 private은 지역 클래스 분포만 더한다 | 학습 없는 "공유 + log 지역 비율"이 FML/FedAPEN 계열 앙상블과 같거나 낫다는 진단, 이득의 공유·private·분포 분해 | 사후 log-prior 보정(Menon 2021, Balanced Softmax), FL prior 보정(FedLC, FedRS, ReBaFL), "개인 head ≈ 클래스 분포"(FedRoD, FedTHE), 균형 전역 + prior 보정 개인화(Xu & Huang 2023) |
| 설계: 작은 동종 모델만 연합, 이기종 private은 연합 뒤 한 번 학습 + 기기 안 cascade | 이 조합 자체는 찾지 못함 | 작은 공유 + 이기종 private(ProxyFL, FML, FedKD, FedType, FedMRL, pFedMoE, FedAPEN; 모두 매 라운드 private 학습), 연합 뒤 로컬 적응 1회(PersFL, Yu 외 2020; 같은 구조만), 기기는 작은 모델만(FedGKT; 큰 모델은 서버), FedAvg-FT·FedRep·FedBABU |
| 효율: 기기 학습 연산·통신·추론 | 같은 프로토콜에서 이중 모델 방법 대비 연산 절감 실측 | sub-model·동결(HeteroFL, FjORD, FedRolex, SmartFreeze, ProFL), cascade(SplitGP, FedHLM, Dirichlet-UQ PFL), 샘플 선택(FedBalancer, FedFT-EDS) |

**예상 공격과 답**: "그냥 Bayes label-shift 보정"(기여는 기법이 아니라 HtFL 진단), "PersFL의 학생을 이기종으로 바꾼 것"(진단이 설계를 정당화, 효율 실측), "FedAvg-FT와 무엇이 다른가"(scout 단독 = c8 FedAvg + head FT 줄을 표에 명시하고, 연합 뒤 private 학습 vs Local 단독 private 비교).

## 2. 본 표 기준선과 비교 대상

| 연구 | 출처 | URL | 내용 | 우리와의 관계 | 읽음 |
|---|---|---|---|---|---|
| FedProto | AAAI 2022 | https://arxiv.org/abs/2105.00243 | 클래스 prototype 교환 | 구조 간 채널 기준선(관찰 1) | [지식] |
| FedTGP | AAAI 2024 | https://arxiv.org/abs/2401.03230 | 학습 가능한 전역 prototype + 마진 | 구조 간 채널 기준선 | [지식] |
| FedGH | ACM MM 2023 | | 전역 head를 서버에서 학습 | 구조 간 채널 기준선 | [지식] |
| LG-FedAvg | 2020 | https://arxiv.org/abs/2001.01523 | head(상위층)만 평균 | 구조 간 채널 기준선 | [지식] |
| FedSA (semantic anchors) | | | 의미 앵커로 표현 정렬. λ=1이 공식값 | 구조 간 채널 기준선 | [지식] |
| FedRG | | | prototype 그래프 집계 | 구조 간 채널 기준선 | [지식] |
| FedAPEN | KDD 2023 | DOI 10.1145/3580305.3599344, 코드 https://github.com/zhenqincn/FedAPEN | 공유 + private logit 적응 앙상블, 상호 학습. 공식 코드 lr 0.01(private·공유 공통), 감쇠 0.99, local epoch 5, 100R | 공유 모델 기준선. 관찰 2의 주 대상 | [본문(코드)] |
| pFedMoE | 2024 | https://arxiv.org/abs/2402.01350 | 공유 extractor와 private extractor를 게이트로 혼합. lr 그리드 {0.001, 0.01, 0.1, 1}, 두 extractor 같은 lr. 분해 ablation 없음, Standalone 대비 +0.6~2.9 | 공유 모델 기준선. 관찰 1 근거 | [본문] |
| FedMRL | NeurIPS 2024 | https://arxiv.org/abs/2406.00488 | 작은 동종 공유 모델 + Matryoshka 표현 융합. SGD lr 0.01. Standalone 대비 +1.8(CIFAR-100, N=10) | 공유 모델 기준선. 관찰 1 근거 | [본문] |
| FML | 2020 | https://arxiv.org/abs/2006.16765 | 공유 + private 상호 학습 | 이중 모델 계열 원형 | [요약] |
| FedKD | Nature Comms 2022 | https://arxiv.org/abs/2108.13323 | 큰 local teacher + 작은 공유 student, student만 통신. 둘 다 매 라운드 학습 | 이중 모델. 통신만 줄고 연산은 안 줄음 | [초록] |

## 3. 작은 공유 모델 + 이기종 private (이중 모델 계열)

| 연구 | 출처 | URL | 내용 | 판정 | 읽음 |
|---|---|---|---|---|---|
| ProxyFL | Nature Comms 2023 | https://arxiv.org/abs/2111.11343 | private + 작은 proxy를 DP 상호 학습. Algorithm 1에서 private을 매 라운드 학습 | 부분 겹침 (모델 배치 같음, 학습 일정 다름) | [본문] |
| FedType | ICML 2024 | https://arxiv.org/abs/2407.03247 | 큰 private + 작은 proxy, conformal 불확실성으로 가중한 양방향 KD. 두 모델 매 라운드 갱신, 추론 cascade 없음 | 부분 겹침 | [본문 대부분] |
| Weak client on-device KD (Lim 외) | arXiv 2025 | https://arxiv.org/abs/2503.11151 | 작은 보조 모델은 모두 평균, 강한 클라이언트만 큰 전역 모델을 증류(매 라운드) | 부분 겹침 | [본문] |
| FedRes (Federated Residual Learning) | arXiv 2020 | https://arxiv.org/abs/2003.12880 | 전역 + 로컬 출력의 합으로 예측, 서로의 잔차 학습(함께 학습) | 부분 겹침 (logit 잔차만). 이전 세션 검토: 방향은 같지만 방법은 다름 | [요약] |
| KOALA | arXiv 2024 | https://arxiv.org/abs/2407.05268 | 작은-큰 모델 간 증류/2단계, 큰 모델을 연합 중 학습 | 다름 | [요약] |
| FedTSA | arXiv 2024 | https://arxiv.org/abs/2407.05098 | 2단계 구조, 큰 모델을 연합 중 학습 | 다름 | [요약] |
| FedMKT | arXiv 2024 | https://arxiv.org/abs/2406.02224 | 큰/작은 언어 모델 상호 지식 전이 | 다름 (참고) | [요약] |

## 4. 연합 뒤 로컬 적응 / 2단계

| 연구 | 출처 | URL | 내용 | 판정 | 읽음 |
|---|---|---|---|---|---|
| **PersFL** | arXiv 2021 | https://arxiv.org/abs/2105.15191 | 1단계 FedAvg만, 2단계에서 각 클라이언트가 최적 전역 스냅샷을 teacher로 로컬 KD 1회. student는 teacher로 초기화(같은 구조), 연산 절감 주장 없음 | **강한 부분 겹침** (2단계 골격). 이기종·잔차·head 개인화 없음 | [본문] |
| Salvaging FL by Local Adaptation (Yu, Bagdasaryan, Shmatikov) | arXiv 2020 | https://arxiv.org/abs/2002.04758 | FL 뒤 로컬 FT, base 동결 후 top 재학습, 같은 구조 KD | 부분 겹침 | [본문] |
| FedAvg-FT, FedRep, FedBABU | FedRep ICML 2021, FedBABU ICLR 2022 | https://arxiv.org/abs/2102.07078 , https://arxiv.org/abs/2106.06042 | 전역 학습 후 head 개인화 / 개인 head / 고정 head 학습 후 FT | 우리 shareall + 마지막 개인화와 같은 패턴. "FedAvg-FT와 무엇이 다른가" 대비 | [지식] (이번에 미확인) |

## 5. 라벨 분포 보정과 개인화 head (관찰 2 관련)

| 연구 | 출처 | URL | 내용 | 판정 | 읽음 |
|---|---|---|---|---|---|
| FedRoD | ICLR 2022 | https://arxiv.org/abs/2107.00778 | generic head(BSM) + personal head를 logit에서 합침. hyper 변형은 클래스 분포만으로 personal head 생성 → "personal head ≈ 분포의 함수" 암시. 사후 log-prior 기준선은 없음 | 부분 겹침 (취지상 가장 가까움) | [본문] |
| Xu & Huang, Joint Training-Calibration for Test-Time Personalization | ICML 2023 FL WS / CIKM 2023 | https://mlanthology.org/icmlw/2023/xu2023icmlw-joint | 균형 전역 모델 + prior 추정 + 예측 보정으로 개인화(같은 구조) | 부분 겹침 (메커니즘상 가장 가까움) | [초록] |
| FedTHE | ICLR 2023 | https://arxiv.org/abs/2205.10920 | global head + personal head 앙상블, personal head는 치우친 분포에 과적합 | 부분 겹침 | [요약] |
| FedLC | ICML 2022 | https://proceedings.mlr.press/v162/zhang22p.html | 학습 단계 pairwise margin 보정, 전역 정확도만 평가 | 다름 (인용 필수) | [본문] |
| FedRS | KDD 2021 | | 없는 클래스의 softmax 갱신 제한 | 다름 (인용) | [요약] |
| ReBaFL | arXiv 2023 | https://arxiv.org/abs/2303.06314 | 완화된 prior-calibrated softmax. 로컬 prior가 0인 클래스에서 보정이 불안정하다고 지적(우리 log prior smoothing과 같은 문제) | 다름 (인용) | [본문] |
| Saerens, Latinne, Decaestecker | Neural Computation 2002 | DOI 10.1162/089976602753284446 | 새 클래스 비율에 맞춰 분류기 출력을 사후 보정(EM으로 비율 추정) | 명제 1의 원조 (인용 필수) | [초록] |
| BBSE (Lipton, Wang, Smola) | ICML 2018 | https://arxiv.org/abs/1802.03916 | 블랙박스 예측기로 label shift 탐지·보정 | 명제 1 관련 (인용 필수) | [초록] |
| Logit adjustment (Menon 외) | ICLR 2021 | https://arxiv.org/abs/2007.07314 | 사후 log-prior 보정의 원조 | 기법 원조 (인용 필수) | [지식] |
| Balanced Softmax (Ren 외) | NeurIPS 2020 | https://arxiv.org/abs/2007.10740 | 클래스 prior 보정 softmax | 기법 원조 (인용 필수) | [지식] |

## 6. 기기 연산 절감

| 연구 | 출처 | URL | 내용 | 판정 | 읽음 |
|---|---|---|---|---|---|
| FedGKT | NeurIPS 2020 | https://arxiv.org/abs/2007.14513 | 기기는 작은 CNN만, 큰 모델은 서버가 KD로 학습. 기기 FLOPs 9~17배 감소 | 부분 겹침 ("기기는 작은 모델만" 논리가 가장 가까움, 큰 모델이 서버) | [초록] |
| HeteroFL | ICLR 2021 | https://arxiv.org/abs/2010.01264 | 폭을 줄인 sub-model을 매 라운드 학습 | 다름 (별도 private 없음) | [지식] |
| FjORD | NeurIPS 2021 | | ordered dropout sub-model | 다름 | [지식] |
| FedRolex | NeurIPS 2022 | https://arxiv.org/abs/2212.01548 | 롤링 sub-model 추출 | 다름 | [요약] |
| SmartFreeze | arXiv 2024 | https://arxiv.org/abs/2408.09101 | 블록 단위 점진 학습·동결 | 다름 (모델 하나) | [요약] |
| ProFL | arXiv 2024 | https://arxiv.org/abs/2404.13349 | 점진 학습으로 메모리 절감 | 다름 | [요약] |
| FedFreeze | FGCS 2025 | | 층 동결로 연산 절감 | 다름 | [요약] |

## 7. cascade / early exit / 샘플 선택

| 연구 | 출처 | URL | 내용 | 판정 | 읽음 |
|---|---|---|---|---|---|
| SplitGP | INFOCOM 2023 | https://arxiv.org/abs/2212.08343 | client 개인화부 + 서버 일반화부, 추론 때 확신도로 서버에 넘김(방향 반대, split learning) | 부분 겹침 | [초록] (threshold는 요약) |
| FedHLM | arXiv 2025 | https://arxiv.org/abs/2507.00082 | 기기 SLM이 불확실할 때만 서버 LLM | 부분 겹침 (큰 모델이 서버) | [요약] |
| Dirichlet-based UQ for Personalized FL | arXiv 2023 | https://arxiv.org/abs/2312.11230 | 불확실성으로 local/global 선택 | 부분 겹침 (크기 비대칭 동기 없음) | [요약] |
| FedBalancer | MobiSys 2022 | https://arxiv.org/abs/2201.01601 | 정보량 큰 샘플만 학습, time-to-accuracy 1.2~4.5배 | 부분 겹침 (샘플 건너뛰기) | [초록] |
| FedFT-EDS | arXiv 2025 | https://arxiv.org/abs/2501.00170 | entropy 기반 샘플 선택 + 부분 학습 | 부분 겹침 | [요약] |
| Centaur | arXiv 2022 | https://arxiv.org/abs/2211.04175 | 기기 데이터 선택 | 부분 겹침 | [요약] |

## 8. 최근 이기종 FL 지식 공유 형태와 survey

| 연구 | 출처 | URL | 내용 | 관계 | 읽음 |
|---|---|---|---|---|---|
| FedTopo | arXiv 2026 | https://arxiv.org/pdf/2607.26801 | 지식 공유를 파라미터·prototype·증류로 분류, 구조가 다르면 공통 좌표 공간이 유지되지 않는다고 지적하고 표현 간 관계를 공유 | 관찰 1과 같은 문제의식. **본문 확인 필요** | [요약] |
| FedRE | arXiv 2025 | https://arxiv.org/pdf/2511.22265 | 얽힌 표현을 새 형태의 클라이언트 지식으로 | 관련 연구 | [요약] |
| FedIN | arXiv 2023 | https://arxiv.org/pdf/2304.00759 | 공개 데이터 없이 중간층 정렬 | 관련 연구 | [요약] |
| Selective-FD | Nature Comms 2024 | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10774276/ | 연합 증류에서 잘못된 지식 걸러내기 | 관련 연구 | [요약] |
| FedHe | arXiv 2021 | https://arxiv.org/pdf/2110.09910 | 이기종 모델 + 통신 효율 | 관련 연구 | [요약] |
| FedSKD | arXiv 2025 | https://arxiv.org/pdf/2503.18981 | 집계 없는 이기종 FL, 유사도 KD | 관련 연구 | [요약] |
| Heterogeneous FL using Knowledge Codistillation | arXiv 2023 | https://arxiv.org/pdf/2310.02549 | 이기종 공동 증류 | 관련 연구 | [요약] |
| Heterogeneous FL: State-of-the-art and Research Challenges | survey 2023 | https://arxiv.org/pdf/2307.10616 | 이기종 FL survey | 배경 | [요약] |
| MHFL survey | arXiv 2023 | https://arxiv.org/abs/2312.12091 | 모델 이기종 FL survey("사후 학습" 분류 확인 필요) | 배경 | 미확인 |

## 9. 설정·도구

| 항목 | URL | 쓰임 | 읽음 |
|---|---|---|---|
| MLPerf Tiny (Banbury 외 2021) | https://arxiv.org/abs/2106.07597 | scout c8 = MLPerf Tiny 이미지 분류 ResNet-8 크기 근거 | [지식] |
| HtFLlib | | 실험 프레임워크, pool projector(-fd 512) 관례 | [지식] |

## 10. 다음에 본문을 읽을 것 (우선순위)

1. PersFL, ProxyFL, FedType: 2단계·이중 모델과의 차이를 정확히 쓰기 위해
2. FedRoD, Xu & Huang 2023: 관찰 2의 "새로운 부분"을 지키기 위해
3. FedTopo: 관찰 1과 문제의식이 겹침
4. FedGKT, SplitGP: "기기는 작은 모델만", cascade와의 차이
5. FedAvg-FT / FedBABU / FedRep: shareall + 마지막 개인화와의 차이
6. FedGH, FedSA, FedRG, FedProto, FedTGP, LG-FedAvg: 본 표 기준선 정식 인용 정보(학회, URL)
