#!/usr/bin/env bash
# One matched seed-0 screening run for the real legacy FedDSA dual state:
# meta-updated fast anchor plus 0.9999 stable EMA.
# shortcut: seed-0 first, replicate only if it beats the fixed-local control.
#SBATCH --job-name=feddsa_legacy_meta
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=batch_grad
#SBATCH --time=4-00:00:00
#SBATCH --output=/nas2/data/a2019102224/HtFLlib/logs/slurm-%A.out

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

SEED=0
RUN="feddsa_scale5_500_legacy_meta_dual_slurm${SLURM_JOB_ID:-manual}_s${SEED}"

exec "${PYTHON_BIN}" -u main.py \
  -go "${RUN}" \
  -did 0 -dev cuda -data Cifar100 -ncl 100 -m HtFE-img-8 \
  -lbs 128 -lr 0.01 -gr 500 -eg 10 -ls 1 -algo FedDSA -jr 1 -nc 20 \
  -fd 512 -lam 1.0 -mcl 1.0 \
  --shared_classifier_scale 5.0   \
  --anchor_codebook gaussian  --separate_anchor_roles true \
  --new_anchor_mode none --semantic_gate_mode meta \
  --anchor_momentum 0.9999 --meta_anchor_lr 1000.0 --meta_inner_lr 0.01 \
  --agreement_weighting true --mcl_variant fedsa \
  --use_amp false --max_parallel_clients 1  \
  --pin_memory true --random_seed "${SEED}" -sfn "temp_${RUN}"
