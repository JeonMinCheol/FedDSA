#!/usr/bin/env bash
#SBATCH --job-name=feddsa_single_diag
#SBATCH --array=0-14%4
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=batch_grad
#SBATCH --time=4-00:00:00
#SBATCH --output=/nas2/data/a2019102224/HtFLlib/logs/slurm-%A_%a.out

set -euo pipefail

REPO=/nas2/data/a2019102224/HtFLlib
PYTHON_BIN=/nas2/data/a2019102224/miniconda3/envs/htfllib/bin/python
cd "${REPO}/system"

# htfllib's PyTorch build needs the bundled Intel notification shim.  The
# environment's C++ runtime must precede the host runtime for SciPy/sklearn.
if [[ -f /tmp/libittnotify.so ]]; then
  export LD_PRELOAD=/tmp/libittnotify.so
fi
export LD_LIBRARY_PATH=/nas2/data/a2019102224/miniconda3/envs/htfllib/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODE_INDEX=$((TASK_ID / 3))
SEED=$((TASK_ID % 3))
case "${MODE_INDEX}" in
  0) MODE=d0; LAM=1.0; MCL=1.0; AGREEMENT=true; CODEBOOK=gaussian ;;
  1) MODE=d1; LAM=1.0; MCL=0.0; AGREEMENT=true; CODEBOOK=gaussian ;;
  2) MODE=d2; LAM=0.0; MCL=1.0; AGREEMENT=true; CODEBOOK=gaussian ;;
  3) MODE=d3; LAM=1.0; MCL=1.0; AGREEMENT=false; CODEBOOK=gaussian ;;
  4) MODE=d4; LAM=1.0; MCL=1.0; AGREEMENT=true; CODEBOOK=simplex ;;
  *) echo "invalid array task ${TASK_ID}" >&2; exit 2 ;;
esac

RUN="feddsa_scale5_linear_single_${MODE}_100_slurm${SLURM_ARRAY_JOB_ID:-manual}_${SEED}"

exec "${PYTHON_BIN}" -u main.py \
  -go "${RUN}" \
  -did 0 -dev cuda -data Cifar100 -ncl 100 -m HtFE-img-8 \
  -lbs 128 -lr 0.01 -gr 100 -eg 10 -ls 1 -algo FedDSA -jr 1 -nc 20 \
  -fd 512 -lam "${LAM}" -mcl "${MCL}" \
  --shared_classifier_scale 5.0   \
   --anchor_codebook "${CODEBOOK}" --fixed_anchor true \
  --agreement_weighting "${AGREEMENT}" --semantic_gate_mode off \
  --use_amp false --max_parallel_clients 1  \
  --random_seed "${SEED}" -sfn "temp_${RUN}"
