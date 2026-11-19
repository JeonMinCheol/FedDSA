#!/usr/bin/env bash
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=batch_grad
#SBATCH  -w ariel-v4
#SBATCH --time=4-00:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORK_DIR="${SCRIPT_DIR}"
# sbatch는 스크립트를 spool 디렉터리로 복사해 실행하므로 제출 위치에서 main.py를 찾는다.
if [[ ! -f "${DEFAULT_WORK_DIR}/main.py" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  for d in "${SLURM_SUBMIT_DIR}/system" "${SLURM_SUBMIT_DIR}"; do
    [[ -f "${d}/main.py" ]] && { DEFAULT_WORK_DIR="${d}"; break; }
  done
fi
WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_DIR}}"
if [[ ! -f "${WORK_DIR}/main.py" ]]; then
  echo "main.py was not found in WORK_DIR=${WORK_DIR}" >&2
  exit 1
fi
cd "${WORK_DIR}"

mkdir -p logs

# CONDA_ENV="${CONDA_ENV:-mincheol}"
# if [[ -n "${CONDA_ENV}" ]]; then
#   CONDA_SH="${CONDA_SH:-}"
#   if [[ -z "${CONDA_SH}" ]]; then
#     for candidate in \
#       "${HOME}/conda3/etc/profile.d/conda.sh" \
#       "${HOME}/data/conda3/etc/profile.d/conda.sh" \
#       "${WORK_DIR}/../../conda3/etc/profile.d/conda.sh"; do
#       if [[ -f "${candidate}" ]]; then
#         CONDA_SH="${candidate}"
#         break
#       fi
#     done
#   fi
#   if [[ -f "${CONDA_SH}" ]]; then
#     # shellcheck source=/dev/null
#     source "${CONDA_SH}"
#     conda activate "${CONDA_ENV}"
#   else
#     echo "CONDA_ENV=${CONDA_ENV} was set, but conda.sh was not found at ${CONDA_SH}" >&2
#     exit 1
#   fi
# fi

PYTHON_BIN="${PYTHON_BIN:-/nas2/data/n1mb0606/anaconda3/envs/htfllib/bin/python}"
HEAD_MODE="${HEAD_MODE:-original}"
PROJECTOR_MODE="${PROJECTOR_MODE:-linear}"
SHARED_CLASSIFIER_SCALE="${SHARED_CLASSIFIER_SCALE:-5.0}"

case "${HEAD_MODE}" in
  original|linear|cosine) ;;
  *) echo "Invalid HEAD_MODE=${HEAD_MODE}; use original, linear, or cosine" >&2; exit 2 ;;
esac
case "${PROJECTOR_MODE}" in
  original|pool|linear|identity) ;;
  *) echo "Invalid PROJECTOR_MODE=${PROJECTOR_MODE}; use original, pool, linear, or identity" >&2; exit 2 ;;
esac

MODEL_ARGS=(
  
  
  --shared_classifier_scale "${SHARED_CLASSIFIER_SCALE}"
)

DEVICE_ID="${DEVICE_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="${DEVICE_ID}"
echo "[$(date '+%F %T')] host=$(hostname)"
echo "workdir=${WORK_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} device_id=${DEVICE_ID}"
echo "python=$(${PYTHON_BIN} --version 2>&1)"
echo

# LG-FedAvg FedProto FedGH FedSA FedTGP FedRG FedAPEN FedMRL
ALGO="${ALGO:-FedSA}"
DATASET="${DATASET:-Cifar100}"   # Cifar100 | Flowers102 | TinyImagenet
SEED="${SEED:-0}"
GR="${GR:-500}"
EG="${EG:-10}"
MAX_PAR="${MAX_PAR:-4}"
# 알고리즘별 추가 인자. 예: EXTRA_ARGS="--global_arch c20"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

case "${DATASET}" in
  Cifar100)     NCL=100; LBS=128 ;;
  Flowers102)   NCL=102; LBS=32 ;;
  TinyImagenet) NCL=200; LBS=128 ;;
  *) echo "Invalid DATASET=${DATASET}; use Cifar100, Flowers102, or TinyImagenet" >&2; exit 2 ;;
esac

RUN_NAME="${RUN_NAME:-${ALGO}-${DATASET}-s${SEED}}"
export PYTHONPATH="${WORK_DIR}/stylegan/stylegan-utils${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
  "${PYTHON_BIN}" -u main.py
  -t 1 -lr 0.01 -jr 1 -lbs "${LBS}" -ls 1 -nc 20 -ncl "${NCL}" -gr "${GR}" -eg "${EG}"
  -algo "${ALGO}" --max_parallel_clients "${MAX_PAR}"
  --pin_memory true --use_amp false --fedsa_augment false
  -data "${DATASET}" -m HtFE-img-8 -fd 512 -did "${DEVICE_ID}" -slr 0.01 --random_seed "${SEED}"
  -go "${RUN_NAME}" -sfn "temp_${RUN_NAME}"
  --rg_graph_target uniform --node_dim 40 --adjalpha 3 --gc_epoch 10 --gnn_learning_rate 0.001 --gnn_weight_decay 0.0001
  "${MODEL_ARGS[@]}"
  "${EXTRA[@]}"
)

printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n\n'

LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.out}"
if [[ "${LOG_FILE}" != /* ]]; then
  LOG_FILE="${WORK_DIR}/${LOG_FILE}"
fi
mkdir -p "$(dirname -- "${LOG_FILE}")"
echo "log=${LOG_FILE}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # sbatch 안에서는 포그라운드로 실행한다. 배경 실행하면 배치 스크립트 종료와 함께 잡이 정리되어 학습도 죽는다.
  "${CMD[@]}" </dev/null >"${LOG_FILE}" 2>&1
else
  nohup "${CMD[@]}" </dev/null >"${LOG_FILE}" 2>&1 &
  echo "started pid=$!"
fi