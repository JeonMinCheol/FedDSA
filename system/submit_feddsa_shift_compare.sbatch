#!/usr/bin/env bash
# EXPERIMENTAL: matched FedDSA anchor ablations and controls.
# Use MODE=dual_gate or MODE=class_boundary for the dual paths, and MODE=fixed for the control.
# shortcut: seed-0 screen before any replication.
#SBATCH --job-name=feddsa_shift_screen
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=batch_grad
#SBATCH --time=4-00:00:00
#SBATCH --output=/nas2/data/a2019102224/HtFLlib/logs/slurm-%A.out

set -euo pipefail

REPO=/nas2/data/a2019102224/HtFLlib
PYTHON_BIN=/nas2/data/a2019102224/miniconda3/envs/htfllib/bin/python
MODE=${MODE:-utility_single}
DATASET=${DATASET:-Cifar100}
FULL_PARTICIPATION=${FULL_PARTICIPATION:-false}
JOIN_RATIO_OVERRIDE=${JOIN_RATIO_OVERRIDE:-}
MAX_PARALLEL_CLIENTS=${MAX_PARALLEL_CLIENTS:-4}
KEEP_CLIENT_MODELS_ON_GPU=${KEEP_CLIENT_MODELS_ON_GPU:-true}
MCL_GAP_MODE=${MCL_GAP_MODE:-min}
MOVING_ANCHOR_EMA=${MOVING_ANCHOR_EMA:-1.0}
UTILITY_LOCAL_GATE=${UTILITY_LOCAL_GATE:-true}
UTILITY_WEIGHT_NORMALIZE=${UTILITY_WEIGHT_NORMALIZE:-true}
UTILITY_SIGNAL_MODE=${UTILITY_SIGNAL_MODE:-observed}
UTILITY_TEMPERATURE=${UTILITY_TEMPERATURE:-0.1}
UTILITY_EMA=${UTILITY_EMA:-0.25}
UTILITY_MIN_WEIGHT=${UTILITY_MIN_WEIGHT:-0.25}
UTILITY_MAX_WEIGHT=${UTILITY_MAX_WEIGHT:-1.5}
UTILITY_VALIDATION_PROBE=${UTILITY_VALIDATION_PROBE:-false}
EXPERIMENTAL_LINEAR_COSINE_AGREEMENT=${EXPERIMENTAL_LINEAR_COSINE_AGREEMENT:-false}
RELIABILITY_WEIGHTED_AGGREGATION=${RELIABILITY_WEIGHTED_AGGREGATION:-false}
RELIABILITY_AGGREGATION_TEMPERATURE=${RELIABILITY_AGGREGATION_TEMPERATURE:-1.0}
RELIABILITY_AGGREGATION_GEOMETRY=${RELIABILITY_AGGREGATION_GEOMETRY:-euclidean}
ROUNDS=${ROUNDS:-500}
SEED=${SEED_OVERRIDE:-0}
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

if [[ -n "${JOIN_RATIO_OVERRIDE}" ]]; then
  JOIN_RATIO="${JOIN_RATIO_OVERRIDE}"
elif [[ "${FULL_PARTICIPATION}" == "true" ]]; then
  JOIN_RATIO=1.0
else
  JOIN_RATIO=0.5
fi

JOIN_TAG="${JOIN_RATIO//./p}"
RUN="feddsa_scale5_${ROUNDS}_${DATASET}_${MODE}_jr${JOIN_TAG}_slurm${SLURM_JOB_ID:-manual}_s${SEED}"

COMMON=(
  -go "${RUN}" -did 0 -dev cuda -data "${DATASET}" -ncl 100 -m HtFE-img-8
  -lbs 128 -lr 0.01 -gr "${ROUNDS}" -eg 10 -ls 1 -algo FedDSA
  -jr "${JOIN_RATIO}" -nc 20 -fd 512 -lam 1.0 -mcl 1.0 -cc 0.0
  --shared_classifier_scale 5.0 --mcl_gap_mode "${MCL_GAP_MODE}"
  --moving_anchor_ema "${MOVING_ANCHOR_EMA}"
   --mcl_variant fedsa
  --utility_local_gate "${UTILITY_LOCAL_GATE}"
  --utility_weight_normalize "${UTILITY_WEIGHT_NORMALIZE}"
  --utility_signal_mode "${UTILITY_SIGNAL_MODE}"
  --utility_validation_probe "${UTILITY_VALIDATION_PROBE}"
  --use_amp false --max_parallel_clients "${MAX_PARALLEL_CLIENTS}" 
  --keep_client_models_on_gpu "${KEEP_CLIENT_MODELS_ON_GPU}"
  --pin_memory true --random_seed "${SEED}"
  -sfn "temp_${RUN}"
)

if [[ "${EXPERIMENTAL_LINEAR_COSINE_AGREEMENT}" == "true" ]]; then
  COMMON+=(--experimental_linear_cosine_agreement true)
fi

if [[ "${RELIABILITY_WEIGHTED_AGGREGATION}" == "true" ]]; then
  COMMON+=(
    --reliability_weighted_aggregation true
    --reliability_aggregation_temperature "${RELIABILITY_AGGREGATION_TEMPERATURE}"
    --reliability_aggregation_geometry "${RELIABILITY_AGGREGATION_GEOMETRY}"
  )
fi

if [[ "${MODE}" == "dual_gate" ]]; then
  COMMON+=(
     --fixed_anchor false --new_anchor_mode none
    --semantic_gate_mode meta --dual_disagreement_gate true
    --dual_disagreement_temperature 0.05
    --anchor_momentum 0.9999 --fast_anchor_ema 0.001
    --meta_anchor_lr 1000.0 --meta_inner_lr 0.01
  )
elif [[ "${MODE}" == "meta" ]]; then
  COMMON+=(
     --fixed_anchor false --new_anchor_mode none
    --semantic_gate_mode meta --dual_disagreement_gate false
    --anchor_momentum 0.9999 --fast_anchor_ema 0.001
    --meta_anchor_lr 1000.0 --meta_inner_lr 0.01
  )
elif [[ "${MODE}" == "relation_memory" ]]; then
  COMMON+=(
     --fixed_anchor false --new_anchor_mode relation_memory
    --semantic_gate_mode off --dual_disagreement_gate false
    --new_anchor_ema 0.25 --relation_memory_lr 0.25
    --relation_anchor_lr 0.02 --relation_observation_weight 0.1
    --relation_step_scale 0.25
  )
elif [[ "${MODE}" == "semantic_discriminative" ]]; then
  COMMON+=(
     --fixed_anchor false --new_anchor_mode semantic_discriminative
    --semantic_gate_mode off --dual_disagreement_gate false
    --new_anchor_ema 0.25 --discriminative_anchor_lr 0.05
    --discriminative_anchor_reg 0.01 --discriminative_step_scale 0.25
  )
elif [[ "${MODE}" == "observation_validation" ]]; then
  COMMON+=(
     --fixed_anchor false --new_anchor_mode observation_validation
    --semantic_gate_mode off --dual_disagreement_gate false
    --new_anchor_ema 0.25 --validation_anchor_lr 0.05
    --validation_step_scale 0.25
  )
elif [[ "${MODE}" == "meta_separated" ]]; then
  COMMON+=(
     --fixed_anchor false --new_anchor_mode meta_separated
    --semantic_gate_mode meta --dual_disagreement_gate false
    --new_anchor_ema 0.25 --meta_anchor_lr 1000.0 --meta_inner_lr 0.01
    --meta_step_scale 0.25
  )
elif [[ "${MODE}" == "reliable_single" ]]; then
  COMMON+=(
     --fixed_anchor false --new_anchor_mode none
    --semantic_gate_mode off --dual_disagreement_gate false
    --reliability_weighted_aggregation true
    --reliability_aggregation_temperature 1.0
  )
elif [[ "${MODE}" == "utility_single" ]]; then
  COMMON+=(
     --fixed_anchor false --new_anchor_mode none
    --semantic_gate_mode off --dual_disagreement_gate false
    --prototype_utility true --utility_inner_lr 0.01
    --utility_temperature "${UTILITY_TEMPERATURE}" --utility_ema "${UTILITY_EMA}"
    --utility_min_weight "${UTILITY_MIN_WEIGHT}" --utility_max_weight "${UTILITY_MAX_WEIGHT}"
  )
elif [[ "${MODE}" == "utility_strict_single" ]]; then
  COMMON+=(
     --fixed_anchor false --new_anchor_mode none
    --semantic_gate_mode off --dual_disagreement_gate false
    --prototype_utility true --utility_inner_lr 0.01
    --utility_temperature "${UTILITY_TEMPERATURE}" --utility_ema "${UTILITY_EMA}"
    --utility_min_weight "${UTILITY_MIN_WEIGHT}" --utility_max_weight "${UTILITY_MAX_WEIGHT}"
    --utility_positive_only true
  )
elif [[ "${MODE}" == "single" ]]; then
  COMMON+=(
     --fixed_anchor false --new_anchor_mode none
    --semantic_gate_mode off --dual_disagreement_gate false
  )
elif [[ "${MODE}" == "fixed" ]]; then
  COMMON+=(
     --fixed_anchor true --new_anchor_mode none
    --semantic_gate_mode off --dual_disagreement_gate false
  )
elif [[ "${MODE}" == "boundary" ]]; then
  COMMON+=(
     --fixed_anchor false --new_anchor_mode boundary
    --semantic_gate_mode off --dual_disagreement_gate false
    --new_anchor_ema 0.001 --boundary_anchor_ema 0.001
    --boundary_loss_weight 1.0
  )
elif [[ "${MODE}" == "class_boundary" ]]; then
  COMMON+=(
     --fixed_anchor false --new_anchor_mode class_boundary
    --semantic_gate_mode off --dual_disagreement_gate false
    --new_anchor_ema 0.1 --class_boundary_ema 0.1
    --class_boundary_min 0.1 --class_boundary_max 1.5
    --freeze_semantic_anchor true
    --boundary_loss_weight 1.0
  )
else
  echo "Unknown MODE=${MODE}; expected dual_gate, meta, relation_memory, semantic_discriminative, observation_validation, meta_separated, utility_single, utility_strict_single, reliable_single, single, fixed, boundary, or class_boundary" >&2
  exit 2
fi

exec "${PYTHON_BIN}" -u main.py "${COMMON[@]}"
