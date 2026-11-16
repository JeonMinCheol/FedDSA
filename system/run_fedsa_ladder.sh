#!/usr/bin/env bash
# FedSA -> FedDSA ablation ladder: change one factor per step, same data/seed/rounds.
#   sref FedSA as published (anchor EMA 0.9999)
#   s0 FedSA, fixed anchor (momentum 1.0)
#   s1 + local-head evaluation
#   s2 + trainable Linear projector instead of AdaptiveAvgPool1d
#   s2o/s2p controls: frozen random orthogonal map / pool + trainable Linear(K,K)
#   s3 + MCL gap min instead of max
#   s5 + FedDSA loss weights (-lam 1 -cc 0; later flags override FEDSA_ARGS)
#   s6 + prototypes recollected after training in eval mode
#   s4 FedDSA, fixed anchor (remaining differences: lam/cc/PCW etc.)
# Usage: SEEDS="0" GR=250 DEVICE_ID=0 nohup bash run_fedsa_ladder.sh > logs/ladder.out 2>&1 &
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu35/miniconda3/envs/mincheol/bin/python}"
SEEDS="${SEEDS:-0}"
GR="${GR:-250}"
DEVICE_ID="${DEVICE_ID:-0}"
STEPS="${STEPS:-sref s0 s1 s2 s3 s4}"
FD="${FD:-512}"  # shared feature/anchor dimension K
TAG="${TAG:-ladder_$(date +%Y%m%d)}"
# Settings behind the measured FedSA ~44% / FedDSA fixed ~46% runs.
FEDSA_ARGS="${FEDSA_ARGS:--lam 10 -mcl 1 -cc 1}"
FIXED="--fedsa_anchor_momentum 1.0"
FEDDSA_ARGS="${FEDDSA_ARGS:--lam 1 -mcl 1 -cc 0 --fixed_anchor true}"

COMMON=(-did 0 -dev cuda -data Cifar100 -ncl 100 -m HtFE-img-8 -fd "${FD}"
        -lbs 128 -lr 0.01 -gr "${GR}" -eg 10 -ls 1 -jr 1 -nc 20
        --use_amp false --pin_memory true)

step_args() {
  case "$1" in
    sref) echo "-algo FedSA ${FEDSA_ARGS} --fedsa_anchor_momentum 0.9999" ;;
    s0) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED}" ;;
    s1) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head" ;;
    s2) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --fedsa_projector linear" ;;
    s2o) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector orthofixed" ;;
    s2p) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector pool_linear" ;;
    s3) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --fedsa_projector linear --fedsa_gap_mode min" ;;
    s5) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --fedsa_projector linear --fedsa_gap_mode min -lam 1 -cc 0" ;;
    s6) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --fedsa_projector linear --fedsa_gap_mode min -lam 1 -cc 0 --fedsa_proto_pass eval" ;;
    s4) echo "-algo FedDSA ${FEDDSA_ARGS}" ;;
    *) echo "unknown step $1" >&2; exit 2 ;;
  esac
}

export CUDA_VISIBLE_DEVICES="${DEVICE_ID}"
# local stylegan torch_utils must shadow the unrelated pip package of the same name
export PYTHONPATH="stylegan/stylegan-utils${PYTHONPATH:+:${PYTHONPATH}}"
# many runs share 40 cores; unbounded BLAS/OMP threads oversubscribe the CPU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
for seed in ${SEEDS}; do
  for step in ${STEPS}; do
    run="${TAG}_${step}_s${seed}_r${GR}"; [[ "${FD}" != 512 ]] && run="${run}_fd${FD}"
    read -r -a extra <<< "$(step_args "${step}")"
    cmd=("${PYTHON_BIN}" -u main.py "${COMMON[@]}" "${extra[@]}" -go "${run}" --random_seed "${seed}" -sfn "temp_${run}")
    echo "[$(date '+%F %T')] ${run}"; printf ' %q' "${cmd[@]}"; echo
    "${cmd[@]}" > "logs/${run}.out" 2>&1 || echo "FAILED ${run} (see logs/${run}.out)"
  done
done
