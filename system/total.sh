#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORK_DIR="${SCRIPT_DIR}"
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

PYTHON_BIN="${PYTHON_BIN:-python}"
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

DEVICE_ID="${DEVICE_ID:-${CUDA_VISIBLE_DEVICES:-2}}"
export CUDA_VISIBLE_DEVICES="${DEVICE_ID}"
echo "[$(date '+%F %T')] host=$(hostname)"
echo "workdir=${WORK_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} device_id=${DEVICE_ID}"
echo "python=$(${PYTHON_BIN} --version 2>&1)"
echo

# LG-FedAvg FedProto FedGH FedSA FedTGP FedRG
ALGO="${ALGO:-FedSA}"

CMD=(
  "${PYTHON_BIN}" -u main.py
  -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 128 -ls 1 -nc 20 -ncl 200
  -algo "${ALGO}" --max_parallel_clients 1
  --pin_memory True
  -data TinyImagenet -m HtFE-img-8 -fd 512 -did "${DEVICE_ID} -slr 0.01 --rg_graph_target uniform 
  --node_dim 40 --adjalpha 3 --gc_epoch 10 --gnn_learning_rate 0.001 --gnn_weight_decay 0.0001"
  "${MODEL_ARGS[@]}"
)

printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n\n'

LOG_FILE="${LOG_FILE:-logs/total-TinyImagenet-HtFE-img-8-fd=512-${ALGO}-${HEAD_MODE}.out}"
if [[ "${LOG_FILE}" != /* ]]; then
  LOG_FILE="${WORK_DIR}/${LOG_FILE}"
fi
mkdir -p "$(dirname -- "${LOG_FILE}")"

nohup "${CMD[@]}" </dev/null >"${LOG_FILE}" 2>&1 &
PID=$!
echo "started pid=${PID}"
echo "log=${LOG_FILE}"