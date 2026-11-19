#!/usr/bin/env bash
# GPU 잡 하나에 기준선 run 여러 개(최대 3개)를 같이 돌린다. Slurm grad QOS가 사용자당 GPU 1개라서.
# 규칙: run이 3개면 각 run에 --max_parallel_clients 2.
# 사용 (system/ 에서):
#   sbatch -J pack1 --export=ALL,RUNS="FedAPEN:TinyImagenet:0 FedAPEN:Flowers102:1" submit_baselines_pack.sbatch
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=batch_grad
#SBATCH --time=4-00:00:00
#SBATCH --output=/nas2/data/n1mb0606/FedDSA/system/logs/slurm-%x-%j.out

set -uo pipefail
cd /nas2/data/n1mb0606/FedDSA/system
: "${RUNS:?RUNS 필요 (METHOD:DATASET:SEED 를 공백으로 구분)}"
read -r -a SPECS <<< "${RUNS}"
N=${#SPECS[@]}
(( N >= 1 && N <= 3 )) || { echo "run은 1~3개만" >&2; exit 2; }

PAR=""; (( N == 3 )) && PAR=2
# FedMRL은 스레드 병렬에서 cuDNN 오류와 멈춤이 났다(docs/baselines_progress.md). PACK_PAR=1 로 덮어쓴다.
PAR="${PACK_PAR:-${PAR}}"
export TORCH_NUM_THREADS=$(( 8 / N ))
echo "pack job=${SLURM_JOB_ID:-manual} host=$(hostname) n=${N} max_parallel_clients=${PAR:-default} torch_threads=${TORCH_NUM_THREADS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0 2>/dev/null

PIDS=()
for spec in "${SPECS[@]}"; do
  IFS=: read -r M D S <<< "${spec}"
  METHOD="${M}" DATASET="${D}" SEED="${S}" MAX_PAR="${PAR}" bash submit_baselines.sbatch &
  PIDS+=($!)
  sleep 20   # 데이터 적재가 겹치지 않게
done
RC=0
for p in "${PIDS[@]}"; do wait "${p}" || RC=1; done
exit "${RC}"
