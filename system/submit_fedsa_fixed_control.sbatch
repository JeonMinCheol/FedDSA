#!/usr/bin/env bash
#SBATCH --job-name=fedsa_fixed_control
#SBATCH --array=0-2%3
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

if [[ -f /tmp/libittnotify.so ]]; then
  export LD_PRELOAD=/tmp/libittnotify.so
fi
export LD_LIBRARY_PATH=/nas2/data/a2019102224/miniconda3/envs/htfllib/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=${SLURM_ARRAY_TASK_ID:-0}
RUN="fedsa_scale5_linear_fixed_position_500_slurm${SLURM_ARRAY_JOB_ID:-manual}_s${SEED}"

exec "${PYTHON_BIN}" -u main.py \
  -go "${RUN}" \
  -did 0 -dev cuda -data Cifar100 -ncl 100 -m HtFE-img-8 \
  -lbs 128 -lr 0.01 -gr 500 -eg 10 -ls 1 -algo FedSA -jr 1 -nc 20 \
  -fd 512 -lam 10.0 -mcl 1.0 -cc 1.0 \
  --shared_classifier_scale 5.0   \
  --fedsa_anchor_momentum 0.9999 --fixed_anchor true \
  --use_amp false --max_parallel_clients 20  \
  --pin_memory true --random_seed "${SEED}" -sfn "temp_${RUN}"
