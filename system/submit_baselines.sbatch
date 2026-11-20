#!/usr/bin/env bash
# 기준선 측정 (증강 없음, HtFE-img-8, 20 clients). docs/twin_handoff.md 4절 참고.
# 사용 (system/ 에서):
#   sbatch -J base_FedMRL_Cifar100_s0 --export=ALL,METHOD=FedMRL,DATASET=Cifar100,SEED=0 submit_baselines.sbatch
#   smoke: GR=3 RUN=smoke_FedMRL_Cifar100 을 추가
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=batch_grad
#SBATCH --time=4-00:00:00
#SBATCH --output=/nas2/data/n1mb0606/FedDSA/system/logs/slurm-%x-%j.out

set -euo pipefail

cd /nas2/data/n1mb0606/FedDSA/system
PYTHON_BIN="${PYTHON_BIN:-/nas2/data/n1mb0606/anaconda3/envs/htfllib/bin/python}"  # a2019102224: PYTHON_BIN=... 로 덮어씀
export PYTHONPATH=stylegan/stylegan-utils

: "${METHOD:?METHOD 필요 (FedAPEN|FedAPEN_lr2|FedMRL|Local)}"
: "${DATASET:?DATASET 필요 (Cifar100|Flowers102|TinyImagenet)}"
: "${SEED:?SEED 필요}"
GR_IN="${GR:-}"  # empty = not given (FedScout_sa then defaults to 540, see below)
GR="${GR:-500}"
EG="${EG:-10}"
GARCH="${GARCH:-c20}"   # 공유 모델 구조. 본 표는 c8 (docs/experiment_plan.md 11-19)
NC="${NC:-20}"
JR="${JR:-1}"
RUN="${RUN:-${PREFIX:-base}_${METHOD}_${DATASET}_s${SEED}}"

case "${DATASET}" in
  Cifar100)     DATA_ARGS=(-data Cifar100 -ncl 100 -lbs 128) ;;
  # client-count experiment (experiment_plan 5): 50/100 clients, batch 32; set NC and JR to match
  Cifar100_50|Cifar100_100) DATA_ARGS=(-data "${DATASET}" -ncl 100 -lbs 32)
                [[ "${NC}" == "${DATASET#Cifar100_}" ]] || { echo "NC=${NC} does not match ${DATASET}" >&2; exit 2; } ;;
  Flowers102)   DATA_ARGS=(-data Flowers102 -ncl 102 -lbs 32) ;;
  TinyImagenet) DATA_ARGS=(-data TinyImagenet -ncl 200 -lbs 128) ;;
  *) echo "Invalid DATASET=${DATASET}" >&2; exit 2 ;;
esac

case "${METHOD}" in
  FedAPEN)     METHOD_ARGS=(-algo FedAPEN --global_arch "${GARCH}") ;;
  FedAPEN_lr2) METHOD_ARGS=(-algo FedAPEN --global_arch "${GARCH}" --apen_shared_lr_scale 2) ;;
  # FedAPEN with the shared model's own classifier (64 x C for c8, no pool to -fd): FedAPEN mixes only logits (11-20)
  FedAPEN_nh)  METHOD_ARGS=(-algo FedAPEN --global_arch "${GARCH}" --apen_native_head true) ;;
  # FedScout final config (11-20): c8 scout without projector, realign off, no gate freeze.
  # _sa = head averaged, personalized at R490; _ph = personal head. 100 clients need about 18 GB (fits a 24 GB 3090);
  # on a smaller GPU add --keep_client_models_on_gpu false (models parked in host memory between turns)
  # 11-20 evening, final: _sa adds head lr x8 and the 2-stage schedule (learner trains only after the R490 freeze)
  # 11-20 night: final local stage = 50 learner-only epochs after the R490 freeze (no communication), so 540 rounds
  # in total unless GR is given explicitly (smoke: GR=4 plus --scout_freeze_at 2 still works)
  FedScout_sa) METHOD_ARGS=(-algo FedScout --scout_arch c8 --scout_gate 1000 --scout_realign 0 --scout_projector none
                            --scout_share all --scout_freeze_at 490 --scout_head_lr_scale 8 --scout_learner_after_freeze true)
               [[ -z "${GR_IN}" ]] && GR=540 ;;
  FedScout_ph) METHOD_ARGS=(-algo FedScout --scout_arch c8 --scout_gate 1000 --scout_realign 0 --scout_projector none
                            --scout_share backbone) ;;
  FedMRL)      METHOD_ARGS=(-algo FedMRL --global_arch "${GARCH}") ;;
  Local)       METHOD_ARGS=(-algo FedSA --fedsa_anchor_momentum 1.0 --fedsa_eval head --projector linear
                            --fedsa_anchor_codebook simplex --fedsa_gap_mode min -lam 0 -mcl 0 -cc 0) ;;
  *) echo "Invalid METHOD=${METHOD}" >&2; exit 2 ;;
esac

CMD=("${PYTHON_BIN}" -u main.py
  -did 0 -dev cuda -m HtFE-img-8 -fd 512 -lr 0.01 -gr "${GR}" -eg "${EG}" -ls 1 -jr "${JR}" -nc "${NC}"
  --use_amp false --pin_memory true --fedsa_augment false --random_seed "${SEED}"
  -go "${RUN}" -sfn "temp_${RUN}"
  "${DATA_ARGS[@]}" "${METHOD_ARGS[@]}")
# GPU 하나에 run 3개를 같이 올릴 때만 MAX_PAR=2 (사용자 지시)
# 클라이언트 수 실험(SERIAL, GPU에 run 하나)만 4 (사용자 지시 11-19). 묶음은 cuDNN 오류 이력 때문에 1 유지
[[ -n "${SERIAL:-}" ]] && MAX_PAR=4  # FedScout too (user 11-20); its client threads are first exercised in the ariel smoke
[[ -n "${MAX_PAR:-}" ]] && CMD+=(--max_parallel_clients "${MAX_PAR}")

mkdir -p logs
LOG="logs/${RUN}.out"
# 이전 로그(오류, OOM 등)는 덮어쓰지 않고 보관한다.
if [[ -f "${LOG}" ]]; then
  mv "${LOG}" "logs/${RUN}.prev-$(date +%Y%m%d-%H%M%S).out"
fi

START="$(date '+%F %T')"
GPU="${CUDA_VISIBLE_DEVICES:-unset}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || echo '?')"
echo "run=${RUN} job=${SLURM_JOB_ID:-manual} host=$(hostname) gpu=${GPU} (${GPU_NAME}) start=${START}"
printf 'command:'; printf ' %q' "${CMD[@]}"; printf '\n'

set +e
"${CMD[@]}" </dev/null >"${LOG}" 2>&1
RC=$?
set -e

LEDGER=logs/baselines_ledger.tsv
[[ -f "${LEDGER}" ]] || printf 'run\tjob\thost\tgpu\tstart\tend\texit\tcommand\n' > "${LEDGER}"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${RUN}" "${SLURM_JOB_ID:-manual}" "$(hostname)" "${GPU} ${GPU_NAME}" \
  "${START}" "$(date '+%F %T')" "${RC}" "$(printf '%q ' "${CMD[@]}")" >> "${LEDGER}"
echo "exit=${RC} end=$(date '+%F %T')"
exit "${RC}"
