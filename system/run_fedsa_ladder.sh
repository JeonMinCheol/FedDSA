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
#   s7 = s5 + anchor losses detached from the backbone; s8 = s5 with dimension-aware projector
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

COMMON=(-did 0 -dev cuda -data "${DATA:-Cifar100}" -ncl "${NCL:-100}" -m "${MF:-HtFE-img-8}" -fd "${FD}"
        -lbs 128 -lr 0.01 -gr "${GR}" -eg 10 -ls 1 -jr "${JR:-1}" -nc 20
        --use_amp false --pin_memory true)

step_args() {
  case "$1" in
    sref) echo "-algo FedSA ${FEDSA_ARGS} --fedsa_anchor_momentum 0.9999" ;;
    s0) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED}" ;;
    s1) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head" ;;
    s2) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --fedsa_projector linear" ;;
    s2o) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector orthofixed" ;;
    s2a) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector orthoaffine" ;;
    s2e) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex" ;;
    s3e) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_gap_mode min --fedsa_anchor_codebook simplex" ;;
    s2r) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector ortholearn" ;;
    s2p) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector pool_linear" ;;
    s3) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --fedsa_projector linear --fedsa_gap_mode min" ;;
    s5) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --fedsa_projector linear --fedsa_gap_mode min -lam 1 -cc 0" ;;
    s6) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --fedsa_projector linear --fedsa_gap_mode min -lam 1 -cc 0 --fedsa_proto_pass eval" ;;
    s3w) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_gap_mode min --fedsa_anchor_codebook simplex --fedsa_gap_warmup ${WARMUP:-50}" ;;
    s2eA) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_augment true" ;;
    s3eA) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_augment true" ;;
    s3eK) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_common_model true" ;;
    s3eKL) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_common_model true --fedsa_kd_feat false" ;;
    s3eG) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_group_avg true" ;;
    s3eWD) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_weight_decay ${WD:-5e-4}" ;;
    s3eV) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_second_view stat --fedsa_stat_view_scale ${VSCALE:-1.0}" ;;
    s3eB) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_bottleneck fed --fedsa_bottleneck_beta ${BETA:-1.0}" ;;
    s3eB0) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_bottleneck fixed --fedsa_bottleneck_beta ${BETA:-1.0}" ;;
    s3eP) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_second_view proto_shift --fedsa_stat_view_scale ${VSCALE:-1.0}" ;;
    s3eR) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min -lam 0 -mcl 0 -cc 0 --fedsa_relation ${RELW:-1.0}" ;;
    s3eR0) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min -lam 0 -mcl 0 -cc 0" ;;
    s3ePC) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_second_view proto_shift --fedsa_shift_fixed ${SFIX:-0.45}" ;;
    s3eVC0) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_second_view noise_fixed --fedsa_shift_fixed ${SFIX:-0.45}" ;;
    s3eVK) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_second_view noise_kl --fedsa_stat_view_scale ${VSCALE:-1.0}" ;;
    s3ePN) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_second_view proto_shift --fedsa_shift_norm true --fedsa_stat_view_scale ${VSCALE:-1.0}" ;;
    s3eX) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_peer_head ${PEERW:-1.0}" ;;
    s3eVL) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_second_view local --fedsa_stat_view_scale ${VSCALE:-1.0}" ;;
    s3eVC) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_second_view stat_ce --fedsa_stat_view_scale ${VSCALE:-1.0}" ;;
    s3eVN) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_second_view noise --fedsa_stat_view_scale ${VSCALE:-1.0}" ;;
    s3eH) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_head_avg true" ;;
    s3eS) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_proto_soft ${SOFT:-0.3}" ;;
    s3eHS) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_head_avg true --fedsa_proto_soft ${SOFT:-0.3}" ;;
    s3eA2C) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_augment true --fedsa_second_view ce" ;;
    s3eA2K) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_augment true --fedsa_second_view consist" ;;
    s3eA2P) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_augment true --fedsa_second_view proto" ;;
    s3eA2A) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_augment true --fedsa_second_view anchor" ;;
    s2eM) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_anchor_mixup ${MIXUP:-0.4}" ;;
    s3eM) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_anchor_codebook simplex --fedsa_gap_mode min --fedsa_anchor_mixup ${MIXUP:-0.4}" ;;
    s5e) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --projector linear --fedsa_gap_mode min -lam 1 -cc 0 --fedsa_anchor_codebook simplex" ;;
    s7) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --fedsa_gap_mode min -lam 1 -cc 0 --projector linear --fedsa_align_detach true" ;;
    s8) echo "-algo FedSA ${FEDSA_ARGS} ${FIXED} --fedsa_eval head --fedsa_gap_mode min -lam 1 -cc 0 --projector dimaware" ;;
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
