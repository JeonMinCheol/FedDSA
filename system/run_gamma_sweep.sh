#!/usr/bin/env bash
# FedDSA anchor-update gamma sweep, all modules off (RAAR/CAU/PCW off, class cap off).
# gamma=0 means a fixed anchor (--fixed_anchor true). Jobs are spread round-robin over GPUS;
# each GPU runs its share one after another.
# Usage: GAMMAS="0 0.001 0.01" SEEDS="0 1 2" GPUS="0 1 2 3 4" nohup bash run_gamma_sweep.sh > logs/gamma_sweep.out 2>&1 &
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu35/miniconda3/envs/mincheol/bin/python}"
GAMMAS="${GAMMAS:-0 0.001 0.01}"
SEEDS="${SEEDS:-0 1 2}"
GPUS="${GPUS:-0 1 2 3 4}"
GR="${GR:-500}"
TAG="${TAG:-gsweep_$(date +%Y%m%d)}"

COMMON=(-algo FedDSA -did 0 -dev cuda -data Cifar100 -ncl 100 -m HtFE-img-8 -fd 512
        -lbs 128 -lr 0.01 -gr "${GR}" -eg 10 -ls 1 -jr 1 -nc 20
        -lam 1 -mcl 1 -cc 0 --mcl_gap_mode min --mcl_gap_class_cap false
        --agreement_weighting false --disable_prototype_trust true --local_marginal_cau false
        --anchor_momentum 0.9999 --use_amp false --pin_memory true)

# local stylegan torch_utils must shadow the unrelated pip package of the same name
export PYTHONPATH="stylegan/stylegan-utils${PYTHONPATH:+:${PYTHONPATH}}"
# many runs share 40 cores; unbounded BLAS/OMP threads oversubscribe the CPU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

run_one() {  # gpu gamma seed
  local gpu=$1 gamma=$2 seed=$3 extra run
  if [[ "${gamma}" == "0" ]]; then extra=(--fixed_anchor true); else extra=(--anchor_candidate_mix "${gamma}"); fi
  run="${TAG}_g${gamma}_s${seed}_r${GR}"
  local cmd=("${PYTHON_BIN}" -u main.py "${COMMON[@]}" "${extra[@]}" -go "${run}" --random_seed "${seed}" -sfn "temp_${run}")
  echo "[$(date '+%F %T')] gpu=${gpu} ${run}"; printf ' %q' "${cmd[@]}"; echo
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "logs/${run}.out" 2>&1 || echo "FAILED ${run} (see logs/${run}.out)"
}

read -r -a gpus <<< "${GPUS}"
jobs=()
for seed in ${SEEDS}; do for gamma in ${GAMMAS}; do jobs+=("${gamma} ${seed}"); done; done

for i in "${!gpus[@]}"; do
  (
    for ((j = i; j < ${#jobs[@]}; j += ${#gpus[@]})); do
      read -r gamma seed <<< "${jobs[j]}"
      run_one "${gpus[i]}" "${gamma}" "${seed}"
    done
  ) &
done
wait
echo "[$(date '+%F %T')] sweep done"
