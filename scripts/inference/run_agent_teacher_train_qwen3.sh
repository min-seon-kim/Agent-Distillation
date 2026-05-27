#!/bin/bash
# ===========================================================
# Generate teacher agent trajectories using Qwen3-32B
# Output is saved locally and used to train the student.
#
# Usage:
#   bash scripts/inference/run_agent_teacher_train_qwen3.sh [--use-prefix]
#
# After this completes, run:
#   bash scripts/training/train_agent_qwen3.sh qwen3
# ===========================================================

# ---- Activate project venv (contains vllm) ----------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$SCRIPT_DIR/../../keys/.venv/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
  source "$VENV_ACTIVATE"
  echo "✅ Activated venv: $VENV_ACTIVATE"
else
  echo "⚠️  venv not found at $VENV_ACTIVATE, proceeding with current PATH"
fi

# ===================== User Setting ===================== #
BASE_MODEL="Qwen/Qwen3-32B"
EXP_TYPE="agent"
PORT_BASE=8000
MAX_TOKENS=1024

VLLM_GPU_DEVICES="0,1"
VLLM_TENSOR_PARALLEL_SIZE=2
RETRIEVER_GPU_DEVICES="4"
RETRIEVER_LOG="retriever_server.log"

# Train split datasets (same as original paper)
declare -A DATASETS=(
  ["hotpotqa"]="data_processor/qa_dataset/train/hotpotqa_1000_20250402.json"
  ["math"]="data_processor/math_dataset/train/math_1000_20250414.json"
  ["math2"]="data_processor/math_dataset/train/math_medium_1000_20250430.json"
)

# Prefix memory paths (only used with --use-prefix, generated separately)
declare -A PREFIXS=(
  ["hotpotqa"]="logs/qa_results/vllm/Qwen_Qwen3-32B/hotpotqa_1000_20250402_train/prefix_memory/Qwen3-32B_temp=0.0_seed=42_type=reasoning.json"
  ["math"]="logs/qa_results/vllm/Qwen_Qwen3-32B/math_1000_20250414_train/prefix_memory/Qwen3-32B_temp=0.0_seed=42_type=reasoning.json"
  ["math2"]="logs/qa_results/vllm/Qwen_Qwen3-32B/math_medium_1000_20250430_train/prefix_memory/Qwen3-32B_temp=0.5_seed=42_type=reasoning.json"
)
# ===================================================== #

USE_PREFIX=false
for arg in "$@"; do
  case $arg in
    --use-prefix) USE_PREFIX=true ;;
  esac
done

PIDS=()

cleanup() {
  echo ""
  echo "🧹 Cleaning up servers..."
  kill ${PIDS[*]} 2>/dev/null
  ps -u $USER -o pid,command | grep 'vllm serve' | grep -v grep | awk '{print $1}' | xargs kill 2>/dev/null || true
  pgrep -f 'retriever_server.py' | xargs -r kill
  wait
  echo "✅ All servers stopped."
}

trap 'echo ""; echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM
export VLLM_USE_V1=0
# /dev/shm is limited to 64MB in this container (cannot remount without root).
# NCCL needs >31MB per segment for TP=2, so disable SHM transport and use
# NVLink P2P instead (GPU 0<->1 are NV4-connected, so this is faster anyway).
export NCCL_SHM_DISABLE=1

# stale NCCL shared memory files from previous failed runs can fill /dev/shm
# (64MB limit) and cause "NCCL error: unhandled system error" on next launch.
STALE=$(ls /dev/shm/nccl-* 2>/dev/null | wc -l)
if [ "$STALE" -gt 0 ]; then
  echo "🧹 Removing $STALE stale NCCL shm files from /dev/shm..."
  rm -f /dev/shm/nccl-*
fi

# ---- Retriever server (start first so it's ready by the time vLLM loads) ----
echo "🔍 Launching retriever server..."
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_DEVICES \
  python search/retriever_server.py > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!
PIDS+=($RETRIEVER_PID)
echo "🛰️  Retriever started (PID: $RETRIEVER_PID, GPUs: $RETRIEVER_GPU_DEVICES)"

# ---- vLLM server (tensor-parallel across dedicated GPUs for 32B model) ------
LOG_FILE="vllm.log"
# Pass NCCL env vars inline to ensure they reach the subprocess even when
# the shell's exported environment is filtered.  NCCL_P2P_LEVEL=NVL forces
# GPU-to-GPU traffic over NVLink instead of SHM, so no /dev/shm segment is
# needed even though NCCL_SHM_DISABLE=1 alone is sometimes ignored by
# newer NCCL (2.26+).
NCCL_SHM_DISABLE=1 NCCL_P2P_LEVEL=NVL \
CUDA_VISIBLE_DEVICES=$VLLM_GPU_DEVICES python serve_vllm.py \
  --model "$BASE_MODEL" \
  --tensor-parallel-size $VLLM_TENSOR_PARALLEL_SIZE \
  --port $PORT_BASE \
  --reasoning-parser qwen3 \
  > "$LOG_FILE" 2>&1 &
PIDS+=($!)
echo "📺 Started Qwen3-32B on GPUs $VLLM_GPU_DEVICES with TP=$VLLM_TENSOR_PARALLEL_SIZE (port $PORT_BASE), waiting for startup..."

( tail -n 0 -f "$LOG_FILE" & ) | while read line; do
  echo "$line"
  if [[ "$line" == *"Application startup complete."* ]]; then
    echo "✅ vLLM fully started."
    break
  fi
done

# ---- Generate trajectories --------------------------------------------------
RUN_EXIT_CODE=0
for dataset in "${!DATASETS[@]}"; do
  echo "🧠 Generating trajectories: $dataset"
  AGENT_CMD="python -m exps_research.unified_framework.run_experiment \
    --experiment_type \"$EXP_TYPE\" \
    --data_path \"${DATASETS[$dataset]}\" \
    --model_type vllm \
    --model_id \"$BASE_MODEL\" \
    --max_tokens $MAX_TOKENS \
    --multithreading --use_process_pool --use_single_endpoint \
    --n 1 --temperature 0.0 --top_p 0.8 \
    --seed 42 \
    --verbose \
    --do_filtering"

  if [ "$USE_PREFIX" = true ]; then
    AGENT_CMD="$AGENT_CMD --prefix_memory \"${PREFIXS[$dataset]}\""
  fi

  eval $AGENT_CMD
  RUN_EXIT_CODE=$?
  if [ $RUN_EXIT_CODE -ne 0 ]; then
    echo "⚠️ Failed on dataset: $dataset"
    break
  fi
done

cleanup

if [ $RUN_EXIT_CODE -ne 0 ]; then
  exit $RUN_EXIT_CODE
fi

echo ""
echo "✅ Trajectory generation complete."
echo ""
echo "📂 Filtered trajectory files saved at:"
MODEL_SHORT="Qwen3-32B"
for dataset in "${!DATASETS[@]}"; do
  ds_name=$(basename "${DATASETS[$dataset]}" .json)
  echo "   logs/qa_results/vllm/Qwen_Qwen3-32B/${ds_name}_train/filtered_data/${MODEL_SHORT}_temp=0.0_seed=42_type=agent_filtered.jsonl"
done
echo ""
echo "▶ Next step: bash scripts/training/train_agent_qwen3.sh qwen3"
