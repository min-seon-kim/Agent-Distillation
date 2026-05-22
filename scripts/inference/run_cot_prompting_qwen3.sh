#!/bin/bash
# ===========================================================
# CoT Prompting baseline with Qwen3-1.7B  (no fine-tuning)
# Corresponds to "CoT Prompting" bar in Figure 1.
#
# Usage:
#   bash scripts/inference/run_cot_prompting_qwen3.sh
#
# GPU layout (adjust CUDA_VISIBLE_DEVICES_LIST if needed):
#   Server on GPU 1 → port 8000  (worker 0)
#   Server on GPU 2 → port 8001  (worker 1)
#   Server on GPU 3 → port 8002  (worker 2)
#   → 3 parallel workers, GPU 0 left free
# ===========================================================

# ===================== User Setting ===================== #
BASE_MODEL="Qwen/Qwen3-1.7B"
EXP_TYPE="reasoning"
PORT_BASE=8000
GPU_MEMORY_UTILIZATION=0.6
MAX_TOKENS=4096
PARALLEL_WORKERS=5        # must equal the number of servers below
# GPUs to use (one per worker, in order)
GPUS=(1 2 3 4 5)

declare -A DATASETS=(
  ["hotpotqa"]="data_processor/qa_dataset/test/hotpotqa_500_20250422.json"
  ["math"]="data_processor/math_dataset/test/math_500_20250414.json"
  ["aime"]="data_processor/math_dataset/test/aime_90_20250504.json"
  ["musique"]="data_processor/qa_dataset/test/musique_500_20250504.json"
  ["bamboogle"]="data_processor/qa_dataset/test/bamboogle_125_20250507.json"
  ["gsm"]="data_processor/math_dataset/test/gsm_hard_500_20250507.json"
  ["2wiki"]="data_processor/qa_dataset/test/2wikimultihopqa_500_20250511.json"
  ["olymath"]="data_processor/math_dataset/test/olymath_200_20250511.json"
)
# ===================================================== #

PIDS=()

cleanup() {
  echo ""
  echo "🧹 Cleaning up vLLM servers..."
  kill ${PIDS[*]} 2>/dev/null
  ps -u $USER -o pid,command | grep 'vllm serve' | grep -v grep | awk '{print $1}' | xargs kill 2>/dev/null || true
  wait
  echo "✅ All vLLM servers stopped."
}

trap 'echo ""; echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM
export VLLM_USE_V1=0

# ---- Start vLLM servers -----------------------------------------------
# Key: port = PORT_BASE + worker_id  so worker i connects to its own server.
# GPU index is decoupled from port: GPUS[i] drives the hardware, port is logical.
NUM_SERVERS=${#GPUS[@]}
LOG_FILE=""

for (( idx=0; idx<NUM_SERVERS; idx++ )); do
  gpu=${GPUS[$idx]}
  port=$((PORT_BASE + idx))
  LOG_FILE="vllm_gpu${gpu}.log"

  CUDA_VISIBLE_DEVICES=$gpu python serve_vllm.py \
    --model "$BASE_MODEL" \
    --port $port \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --reasoning-parser qwen3 \
    > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
  echo "🚀 Started vLLM on GPU $gpu → port $port (worker $idx)"
done

# Wait for the last server to finish starting
echo "📺 Waiting for last server to be ready (GPU ${GPUS[-1]}, log: $LOG_FILE)..."
( tail -n 0 -f "$LOG_FILE" & ) | while read line; do
  echo "$line"
  if [[ "$line" == *"Application startup complete."* ]]; then
    echo "✅ vLLM fully started, launching CoT Prompting experiments!"
    break
  fi
done

# ---- Run experiments -----------------------------------------------
RUN_EXIT_CODE=0
for dataset in "${!DATASETS[@]}"; do
  echo "🧠 Running CoT Prompting on: $dataset"
  python -m exps_research.unified_framework.run_experiment \
    --experiment_type "$EXP_TYPE" \
    --data_path "${DATASETS[$dataset]}" \
    --model_type vllm \
    --model_id "$BASE_MODEL" \
    --max_tokens $MAX_TOKENS \
    --multithreading \
    --use_process_pool \
    --parallel_workers $PARALLEL_WORKERS \
    --n 1 --temperature 0.0 --top_p 0.8 \
    --seed 42 \
    --verbose
  RUN_EXIT_CODE=$?
  if [ $RUN_EXIT_CODE -ne 0 ]; then
    echo "⚠️ Experiment failed on dataset: $dataset"
    break
  fi
done

cleanup

if [ $RUN_EXIT_CODE -ne 0 ]; then
  echo "⚠️ Script failed with exit code $RUN_EXIT_CODE"
  exit $RUN_EXIT_CODE
else
  echo "✅ CoT Prompting evaluation completed."
  exit 0
fi
