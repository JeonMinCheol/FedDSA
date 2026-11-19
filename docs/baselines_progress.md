# 기준선 측정 진행 기록 (2026-11-18 ~)

최종 결과는 끝난 뒤 `docs/baselines_<날짜>.md`에 정리한다. 이 문서는 진행 중인 상태와 사건을 기록한다.

## 범위 (지시서 기준)
증강 없음, HtFE-img-8, 20 clients, 500R, 10라운드마다 평가, seed 0·1·2.
집계: 라운드별 seed 3개 평균 곡선의 최고값. 표준편차는 그 라운드의 seed 간 값.

| 기준선 | 데이터셋 | 방법 인자 |
|---|---|---|
| FedAPEN | Flowers102, Tiny, CIFAR-100 (CIFAR은 사용자 요청으로 추가. 다른 노드 run과 겹칠 수 있음) | `-algo FedAPEN --global_arch c20` |
| FedMRL | CIFAR-100, Flowers102, Tiny | `-algo FedMRL --global_arch c20` (smoke 먼저) |
| Local | CIFAR-100 | `-algo FedSA --fedsa_anchor_momentum 1.0 --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min -lam 0 -mcl 0 -cc 0` |
| FedAPEN 공유 lr ×2 | CIFAR-100 | 위 FedAPEN + `--apen_shared_lr_scale 2` |

## 실행 환경과 방식
- 코드: 브랜치 fedsa-feddsa-ladder, 커밋 db47aeb. `utils/test_item_store.py`를 컴퓨트 노드에서 실행해 `ok` 확인(메모리 저장소 버그 수정 반영).
- python: `/nas2/data/n1mb0606/anaconda3/envs/htfllib/bin/python`.
- Slurm grad QOS가 **사용자당 GPU 1개**라서 잡은 한 번에 하나씩만 돈다. 그래서 GPU 잡 하나에 **같은 데이터셋 run 3개를 동시에** 띄운다(사용자 결정).
  - run이 3개면 각 run에 `--max_parallel_clients 2`, `TORCH_NUM_THREADS=2` (사용자 지시). **지시서 기본값(1)과 다르다.** 예전 FedSA 기준선도 `max_parallel_clients=2`였다.
- 스크립트:
  - `system/submit_baselines.sbatch`: run 하나. 지시서 인자 그대로이고, `MAX_PAR`, `GR`, `EG`, `PREFIX`로 일부를 바꿀 수 있다.
  - `system/submit_baselines_pack.sbatch`: `RUNS="METHOD:DATASET:SEED ..."`를 한 GPU에서 동시에 돌린다.
- 기록:
  - run별 명령, 호스트, GPU, 시작·종료 시각, 종료 코드: `system/logs/baselines_ledger.tsv`
  - run 로그: `system/logs/<run>.out`
  - 묶음 로그: `system/logs/slurm-<잡이름>-<잡ID>.out`
  - 다시 띄울 때 이전 로그는 `<run>.prev-<시각>.out`으로 보관한다.

## 대기열 (2026-11-18 14:50 기준, 위에서부터 차례로 실행, 모두 GPU당 3개 묶음 + `max_parallel_clients 1`)

| 잡 ID | 잡 이름 | run | 상태 |
|---|---|---|---|
| 492602 | pack_FedAPEN_Flowers102 | base_FedAPEN_Flowers102_s0~2 | 실행 중 (ariel-v5) |
| 492603 | pack_FedAPEN_TinyImagenet | base_FedAPEN_TinyImagenet_s0~2 | 대기 |
| 492604 | pack_FedAPEN_Cifar100 | base_FedAPEN_Cifar100_s0~2 | 대기 |
| 492605 | pack_Local_Cifar100 | base_Local_Cifar100_s0~2 | 취소 (다른 곳에서 실행) |
| 492606 | pack_FedAPEN_lr2_Cifar100 | base_FedAPEN_lr2_Cifar100_s0~2 | 취소 (다른 곳에서 실행) |
| 492607 | pack_FedMRL_Cifar100 | base_FedMRL_Cifar100_s0~2 | 대기 |
| 492608 | pack_FedMRL_Flowers102 | base_FedMRL_Flowers102_s0~2 | 대기 |
| 492609 | pack_FedMRL_TinyImagenet | base_FedMRL_TinyImagenet_s0~2 | 대기 |

## 사건 기록
- **14:00 total.sh 수정**: 짝 없는 따옴표 제거, sbatch 안에서 포그라운드 실행, 작업 디렉터리 탐색 수정, 데이터셋별 인자 자동 설정. 기준선은 total.sh 대신 위의 전용 스크립트를 쓴다.
- **FedMRL smoke 1차 (492465, 단독, -eg 10)**: 3라운드 오류 없이 끝남. 라운드당 약 11초. 평가는 R0(1.47%) 한 번뿐이라 학습 뒤 평가는 확인하지 못했다. 그래서 `EG=1`로 다시 돌린다.
- **잡 재구성**: 처음에는 run마다 잡 하나로 냈다(492466~74, 492485~87). QOS 때문에 하나씩만 돌아서 모두 취소했다.
  - 이미 돌던 base_FedAPEN_Flowers102_s0(492466, 8라운드, R10 근처 49.9%)도 취소했다. 로그는 `logs/base_FedAPEN_Flowers102_s0.cancelled-492466.out`.
  - 그 뒤 데이터셋이 섞인 묶음(492491~95)으로 냈다가, 같은 데이터셋끼리 묶으라는 지시로 다시 냈다. Tiny도 3개를 동시에 돌린다(사용자 허용).
- **FedMRL smoke 2차 (492490, 3개 묶음, max_parallel_clients 2)**:
  - Flowers102: R0 1.80%, R1 22.24%, R2 21.90%까지 평가한 뒤 R3 학습 중 `RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED`로 종료(14:01:56, exit 1). 로그는 `logs/smoke_FedMRL_Flowers102_s0.cudnn.out`. 학습이 되는 것은 확인했다.
  - 그 시점 GPU 메모리는 9.6/24.5GB라서 단순 OOM으로는 보이지 않는다. 원인 후보는 (a) `max_parallel_clients 2`(스레드 병렬)와 FedMRL 조합의 스레드 안전성, (b) 일시적인 cuDNN 오류.
  - Tiny와 CIFAR smoke는 계속 실행 중이다.
  - **할 일**: smoke가 끝나면 Flowers FedMRL smoke를 `MAX_PAR=2` 단독과 기본값 단독으로 다시 돌려 원인을 가른다. 같은 문제가 FedAPEN 묶음(`max_parallel_clients 2`)에도 생길 수 있으니 pack_APEN_Flowers 초반 로그를 확인한다.
- **FedMRL smoke 2차 최종 (492490)**:
  - Tiny: R3 9.95%, 정상 종료(14:12, exit 0).
  - CIFAR: R0 평가 뒤 로그가 14:01:38에 멈췄다(Flowers가 죽은 시각과 거의 같음). 38분 동안 진전 없이 있다가 잡 시간 제한(40분)으로 14:40에 취소됐다. 같은 CIFAR를 기본값(1)으로 순서대로 학습한 1차 smoke에서는 정상이었다.
  - 판단: **FedMRL + `max_parallel_clients 2`(ThreadPoolExecutor)가 불안정하다** (추정. 원인 미확정). FedMRL은 매 라운드 `load_item('Server', …)`로 서버 모델을 deepcopy하고, 스레드 두 개에서 동시에 일어난다.
  - ~~FedAPEN은 `run_clients`를 쓰지 않는다~~ → **틀렸다.** FedAPEN 서버는 `serverfml`을 상속하고 `run_clients`(스레드)를 쓴다(아래 14:45 참고).
- **14:40 pack_APEN_Flowers (492505) 시작**, ariel-v5.
- **FedMRL 결정 (사용자)**: GPU당 3개 묶음은 유지하고 FedMRL만 `max_parallel_clients 1`(`PACK_PAR=1`). 묶음 492596~98을 제출했다. Slurm은 제출 순서와 우선순위에 따라 차례로 실행한다.
- **14:45 FedAPEN Flowers 묶음(492505, max_parallel_clients 2)에서도 같은 오류**: s0가 R1 학습 중 `CUDNN_STATUS_EXECUTION_FAILED`로 종료(`serverfml.py:41` → `serverbase.py:154` ThreadPoolExecutor → `clientapen.py:79`). 로그는 `logs/base_FedAPEN_Flowers102_s0.cudnn-492505.out`.
  - 정리: 스레드 병렬 2 + 한 GPU에 여러 프로세스인 조건에서 3번 중 2번(FedMRL smoke, FedAPEN) cuDNN 오류가 났고, 같은 GPU의 다른 run이 멈췄다. 단독 실행 + 기본값 1에서는 문제가 없었다. 원인 코드는 확정하지 못했다.
  - 조치: 잡을 모두 취소(492505의 s1·s2 로그는 `*.cancelled-492505.out`)하고, **모든 묶음을 `max_parallel_clients 1`로** 우선순위 순서대로 다시 제출했다(492602~09). 지시서 기본값과 같은 조건이다.
- **11-19 00:10 사용자 결정**: FedAPEN + 개인 head(3×3), Local 우선순위 조정, FedAPEN lr×2를 개인 head + lr×2로 바꾸는 일은 **다른 곳에서 처리**한다. 이 대기열은 바꾸지 않는다.
- **FedAPEN Flowers102 완료 (492602, 16:33)**: 68.53 ± 1.61 @R490 (seed 평균 곡선 최고값, 잠정). FedAPEN Tiny는 00:08에 R210~220, 41%대.
- **11-19 사용자 지시**: Local(492605)과 FedAPEN lr×2(492606) 묶음을 시작 전에 취소했다. 다른 곳에서 돌린다.

## FedAPEN 분해 (11-19, seed 평균, `APENDiag` 줄)
| | 앙상블 최고 | private 단독 | 공유(c20) 단독 | 앙상블 − 더 나은 단독 | w_private |
|---|---|---|---|---|---|
| Flowers102 | 68.53 @R490 | 52.28 | 47.46 | +16.3 | 0.57 |
| Tiny | 48.18 @R480 (잠정) | 28.91 | 28.16 | +19.3 | 0.52 |

- 이득은 거의 전부 앙상블에서 온다. 공유 모델은 R10에는 8.9/3.4%로 낮지만 500R까지 private 수준으로 오른다. handoff F3의 "붕괴"는 초반 라운드에만 해당한다.
- 앙상블 가중치는 학습 데이터의 held-out 5%로만 학습한다(`clientapen.learn_weight`). 테스트 데이터 누출은 없고, 평가는 eval 모드다.
- **+16~19는 이례적으로 크다 → 클라이언트별 분해가 필요하다.** 이미 끝난 Flowers·Tiny 로그에는 클라이언트별 private·공유 값이 없고(앙상블 per-client만 있음), temp_ 저장소라 모델도 남지 않았다.
  그래서 `serverapen.evaluate`에 `APENDiagClient id:private/shared/w` 줄을 추가했다. FedAPEN CIFAR 묶음(492604)부터 기록된다. Flowers도 필요하면 seed 1개를 다시 돌린다(약 1.7시간).
- 다른 노드용 메모: 우리 결합 이득(쌍둥이 대비 +0.5~0.7)과 대비가 크다. private을 잔차가 아니라 FedAPEN처럼 독립 모델 + mutual KL로 학습하는 변형을 CIFAR 150R seed 0으로 확인해 볼 만하다.
- **11-19 Flowers 클라이언트별 분해용 재실행 제출**: 494655 `diag_FedAPEN_Flowers102_s0`(단독, `max_parallel_clients 1`, 500R). 사용자 지시로 맨 앞으로 올렸다(`scontrol top`은 권한 거부, 대신 FedMRL 묶음 492607~09에 `nice=500`). FedAPEN CIFAR 다음에 실행된다.
- FedAPEN CIFAR(492604)에서 `APENDiagClient` 줄이 정상 출력되는 것을 확인했다. R10에서 이미 클라이언트 6·7은 w_private가 0.06·0.03으로, 거의 공유 모델만 쓴다.

## 11-19 계획 변경: 본 표 기준선은 c8 (`docs/experiment_plan.md`, 커밋 da5ac80)
- `submit_baselines.sbatch`에 `GARCH`(기본 c20)를 추가했다. c8 run 이름은 `c8_<METHOD>_<DATASET>_s<seed>`.
- 494655(diag Flowers c20)는 취소했다. 실제로는 13:05~13:36 ariel-v13에서 약 30분 돌다가 취소됐다(로그 `logs/diag_FedAPEN_Flowers102_s0.out`). 같은 시간 FedMRL c20 CIFAR도 다른 노드에서 돌아 GPU 1개 QOS인데 두 잡이 겹친 것으로 보인다(원인 미확인). c8 FedAPEN run이 `APENDiagClient`를 직접 찍으므로 분해는 c8에서 본다.
- c8 묶음 제출(모두 seed 0~2, `PACK_PAR=1`): 494849 FedAPEN Flowers(13:2x 시작, ariel-g1) → 494850 FedAPEN Tiny → 494851 FedAPEN CIFAR → 494852~54 FedMRL CIFAR·Flowers·Tiny.
- FedMRL c20 묶음(492607~09, 부록용)은 c8 뒤로. 492607은 13:06에 막 시작한 것을 `scontrol requeue`로 되돌렸다.
- 주의: `docs/baseline_c8_instructions.md`는 아직 "기본 c20"이라고 되어 있어 계획 문서와 어긋난다.

## FedAPEN CIFAR-100 c20 완료 (492604, 13:05, 부록용)
- 56.95 ± 0.26 @R420 (seed 평균 곡선 최고, 표준편차는 표본 표준편차).
- R500 APENDiag: private 0.435, shared 0.325, w_private 0.565.
- **클라이언트별 분해 (R500, seed 3개, 구조별 평균; 이득 = 앙상블 − max(private, shared))**

| private 구조 (client id mod 8) | 앙상블 | private | shared | w_private | 이득 |
|---|---|---|---|---|---|
| FedAvgCNN | 59.7 | 52.8 | 30.9 | 0.68 | +6.9 |
| GoogLeNet | 59.4 | 53.6 | 32.9 | 0.64 | +5.8 |
| MobileNetV2 | 56.8 | 43.0 | 34.4 | 0.51 | +13.8 |
| ResNet18 | 56.0 | 43.9 | 36.2 | 0.55 | +11.8 |
| ResNet34 | 51.5 | 38.9 | 29.0 | 0.50 | +12.3 |
| ResNet50 | 52.6 | 38.0 | 32.2 | 0.54 | +14.1 |
| ResNet101 | 53.7 | 39.4 | 33.3 | 0.54 | +14.3 |
| ResNet152 (client 7·15) | 59.8 | 28.0 | 29.1 | 0.50 | **+29.4** |

- 이득은 모든 클라이언트에서 양수(최소 +2.5)라서 몇 명에게만 몰려 있지는 않다. 하지만 **private이 약한 큰 모델일수록 크다.** ResNet152는 private 28%, shared 29%인데 앙상블이 60%다.
- 둘 다 30% 미만인 모델의 logit 합이 60%가 되는 것은 보통의 앙상블로는 설명하기 어렵다. 후보 설명: 공유 모델(평균 head)은 전역 클래스 분포로 예측하고, private 모델은 지역 클래스 분포(Dir 0.1)를 안다. logit을 더하면 공유 모델에 지역 클래스 사전확률을 곱하는 효과가 난다(da5ac80의 scout screening prior와 같은 논리). **아직 검증하지 않았다.** 확인하려면 공유 logit + log(지역 클래스 빈도)만으로 정확도를 재 보면 된다.
- **진단 줄 추가 (11-19)**: `clientapen.test_metrics`에서 `shared logit + log(지역 학습 데이터 클래스 빈도)` 정확도를 같이 잰다(테스트 라벨은 쓰지 않음). `APENDiag ... shared+prior=`, `APENDiagClient id:private/shared/w/shared+prior`로 찍힌다. 이미 돌던 c8 Flowers(494849)에는 없고, c8 Tiny(494850)부터 나온다. 판정: shared+prior가 앙상블에 가까우면 FedAPEN 이득의 대부분은 지역 클래스 분포 보정이다.

## 11-19 a2019102224 노드: pFedMoE c8 (본 표 기준선)
- 노드 역할: **ariel.khu.ac.kr**은 기준선 측정, **163.180.117.63**은 FedScout 실험. 이 계정(a2019102224, 로그인 노드 ariel-master)은 ariel에서 **pFedMoE**를 돌린다.
- 인자는 실험 계획대로 원 논문 방식: `-algo pFedMoE --global_arch c8 --pfedmoe_gate_lr 1.0`. 나머지는 지시서 그대로(HtFE-img-8, 20 clients, 500R, `-eg 10`, 증강 없음, `--max_parallel_clients 1`, Flowers는 `-lbs 32`, 나머지는 128). 묶음 없이 run 하나에 잡 하나(array 0-2 = seed 0~2).
- 코드: 커밋 da5ac80을 `logs/pfedmoe_c8_baselines_20261119/source/`에 복사해 두고 그 복사본으로 실행한다. 그 뒤의 커밋은 이 run에 영향을 주지 않는다.
- python: `/nas2/data/a2019102224/miniconda3/envs/htfllib/bin/python` (`LD_LIBRARY_PATH`에 env의 lib 필요).

| 잡 ID | 데이터셋 | run 이름 | 상태 (11-19 제출) |
|---|---|---|---|
| 494873 | CIFAR-100 | `pfedmoe_c8_cifar_r500_s{0,1,2}_20261119` | s0 실행 중 (ariel-v11), s1·s2 대기 |
| 494874 | Flowers102 | `pfedmoe_c8_flowers_r500_s{0,1,2}_20261119` | 대기 |
| 494875 | Tiny-ImageNet | `pfedmoe_c8_tiny_r500_s{0,1,2}_20261119` | 대기 |

- 제출 스크립트: `logs/pfedmoe_c8_baselines_20261119/<run 묶음>/submit.sbatch`. 잡 번호는 같은 폴더의 `jobs.tsv`에 있다.
- 로그: `logs/experiment_groups/pfedmoe_c8_baselines_20261119/<run 묶음>/seed<s>_job<id>.out`. 게이트가 공유 expert에 주는 평균 가중치가 `pFedMoEDiag` 줄에 찍힌다.
- 이 계정도 GPU 1개 QOS라서 9개가 하나씩 차례로 돈다.
- **apenph 취소 (사용자 지시)**: 이 계정에서 돌던 FedAPEN + 개인 head + 공유 lr×3, c20(`logs/fedapen_personal_head_baselines_20261118`, 잡 493123·493124)을 모두 취소했다. Flowers s0은 약 42분 돌다가 취소됐다(로그 `logs/experiment_groups/fedapen_personal_head_baselines_20261118/apenph_flowers/seed0_job493123.out`). Flowers s1·s2와 Tiny s0~2는 시작하지 않았다. 부록 변형 "FedAPEN + 개인 head + lr×3"는 아직 결과가 없다.

## 11-19 역할 정리 (사용자)
- ariel.khu.ac.kr은 기준선 성능 측정만 한다. FedAPEN 이득 원인(클래스 분포 보정 가설) 검증, apenph 같은 변형·ablation은 **163.180.117.63 담당**.
- 이 노드는 c8 FedAPEN 로그에 찍히는 `APENDiag`/`APENDiagClient`(shared+prior 포함) 값을 결과 문서에 그대로 옮겨 넘기기만 한다. 해석과 추가 실험은 63에서.

## 11-19 ariel(a2019102224): 클라이언트 수·참여율 실험, pFedMoE와 Local (CIFAR-100)
- 설정: 같은 nc의 FedSA 기준선(`logs/fedsa_cifar100_nc*_bs32_*`)과 같음. `-data Cifar100_<NC>`, `-lbs 32`, 500R, `-eg 10`, 증강 없음, seed 0~2, `--keep_client_models_on_gpu true`.
- pFedMoE: `--global_arch c8 --pfedmoe_gate_lr 1.0 --max_parallel_clients 1`. Local: 위 Local 인자 + `--max_parallel_clients 4`(같은 nc FedSA 기준선의 mp4).
- 코드: pFedMoE c8과 같은 복사본(da5ac80). 폴더 `logs/scale_cifar100_pfedmoe_local_20261119/`, 잡 번호 `jobs.tsv`.
- 계정당 제출 잡 수 상한이 20개(`QOSMaxSubmitJobPerUserLimit`, array task 하나가 1개로 셈)라서, 각 설정은 **잡 하나에서 seed 0→1→2를 차례로** 돈다(`submit_seq.sbatch`). GPU도 1개 QOS라서 속도는 같다.
- run 로그: `logs/experiment_groups/scale_cifar100_pfedmoe_local_20261119/<설정>/seed<s>.out`.

| 설정 | pFedMoE 잡 | Local 잡 |
|---|---|---|
| nc50, jr 0.5 | 494939 | 494930 |
| nc100, jr 0.5 | 494940 | 494942 |
| nc100, jr 0.3 | 494941 | 494943 |
| nc100, jr 0.1 | 494929 | 494944 |

- 순서: pFedMoE c8 본 표(494873~75)가 끝난 뒤에 실행된다.

## 11-19 클라이언트 수 실험 (사용자 지시, `experiment_plan.md` 5절)
- CIFAR-100, 500R, 배치 32, c8, seed 0~2. 방법: FedAPEN c8, FedMRL c8 (사용자 선택).
- 설정: 50명 join 0.5 (`Cifar100_50`), 100명 join 0.5·0.3·0.1 (`Cifar100_100`). 예전 원고에는 0.3·0.1 값이 없다.
- `submit_baselines.sbatch`에 `NC`, `JR`(기본 20, 1)과 데이터셋 `Cifar100_50`·`Cifar100_100`(-lbs 32)을 추가했다. NC가 데이터셋 이름과 다르면 바로 종료한다.
- 순서: c8 본 표(FedAPEN·FedMRL) 다음, c20 FedMRL(부록) 앞.
- **사용자 지시: 이 실험은 GPU에 run 하나씩.** smoke(494925)는 취소했다. `submit_baselines_pack.sbatch`에 `SERIAL=1`(같은 잡 안에서 seed 0→1→2 차례로)을 추가하고, (방법, 설정)마다 잡 하나로 냈다. 제출 잡 수 제한(20) 때문에 run마다 잡을 내지 않았다.
- 잡: 494931~34 FedAPEN (50·0.5, 100·0.5, 100·0.3, 100·0.1), 494935~38 FedMRL (같은 순서). run 이름 `c8jr<참여율>_<방법>_Cifar100_<명>_s<seed>`.
