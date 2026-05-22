#!/bin/bash
# ===========================================================
# Agent Distillation evaluation with Qwen3-1.7B  (fine-tuned)
# Corresponds to "Agent Distillation (Ours)" bar in Figure 1.
#
# Usage:
#   bash scripts/inference/run_agent_qwen3.sh [BASE_MODEL] [LORA_PATH]
#
# GPU layout (adjust GPUS array if needed):
#   Server on GPU 1 → port 8000  (worker 0)
#   Server on GPU 2 → port 8001  (worker 1)
#   Retriever         GPU 2,3
#   → GPU 0 left free
# ===========================================================

# ===================== User Setting ===================== #
BASE_MODEL=${1:-"Qwen/Qwen3-1.7B"}
LORA_PATH=${2:-"training_outputs/qwen3-1.7B/agent_baseline_qwen2.5_32B_teacher"}
EXP_TYPE="agent"
PORT_BASE=8000
GPU_MEMORY_UTILIZATION=0.6
MAX_LORA_RANK=64
N=8
TEMP=0.4
MAX_TOKENS=1024
PARALLEL_WORKERS=2        # must equal the number of vLLM servers below
# GPUs used for vLLM (retriever uses its own GPUs below)
GPUS=(1 2)

RETRIEVER_GPU_DEVICES="2,3"
RETRIEVER_LOG="retriever_server.log"

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
  pgrep -f 'retriever_server.py' | xargs -r kill
  wait
  echo "✅ All vLLM servers stopped."
}

trap 'echo ""; echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM
export VLLM_USE_V1=0

# ---- Start retriever server -------------------------------------------------
echo "🔍 Launching retriever server..."
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_DEVICES \
  python search/retriever_server.py \
  > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!
PIDS+=($RETRIEVER_PID)
echo "🛰️  Retriever server started (PID: $RETRIEVER_PID, GPUs: $RETRIEVER_GPU_DEVICES)"

# ---- Start vLLM servers -----------------------------------------------
# port = PORT_BASE + worker_id  →  worker i connects to its own server.
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
    --lora-modules "finetune=$LORA_PATH" \
    --max-lora-rank $MAX_LORA_RANK \
    --reasoning-parser qwen3 \
    > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
  echo "🚀 Started vLLM on GPU $gpu → port $port (worker $idx)"
done

echo "📺 Waiting for last server to be ready (GPU ${GPUS[-1]}, log: $LOG_FILE)..."
( tail -n 0 -f "$LOG_FILE" & ) | while read line; do
  echo "$line"
  if [[ "$line" == *"Application startup complete."* ]]; then
    echo "✅ vLLM fully started, launching Agent Distillation experiments!"
    break
  fi
done

# ---- Run experiments -----------------------------------------------
for dataset in "${!DATASETS[@]}"; do
  echo "🧠 Running Agent Distillation on: $dataset"
  python -m exps_research.unified_framework.run_experiment \
    --experiment_type "$EXP_TYPE" \
    --data_path "${DATASETS[$dataset]}" \
    --model_type vllm \
    --model_id "$BASE_MODEL" \
    --max_tokens $MAX_TOKENS \
    --multithreading \
    --use_process_pool \
    --parallel_workers $PARALLEL_WORKERS \
    --n $N --temperature $TEMP --top_p 0.8 \
    --seed 42 \
    --fine_tuned \
    --lora_folder "$LORA_PATH" \
    --verbose
done

RUN_EXIT_CODE=$?
cleanup

if [ $RUN_EXIT_CODE -ne 0 ]; then
  echo "⚠️ Script failed with exit code $RUN_EXIT_CODE"
  exit $RUN_EXIT_CODE
else
  echo "✅ Agent Distillation evaluation completed."
  exit 0
fi
