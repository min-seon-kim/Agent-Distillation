#!/bin/bash

# ===================== Configuration ===================== #
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

LORA_PATH="agent-distillation/agent_distilled_Qwen2.5-1.5B-Instruct"
MAX_LORA_RANK=64

PORT=8000
GPU_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
# ========================================================= #

# Cleanup handler
cleanup() {
  echo ""
  echo "🧹 Cleaning up retriever and vLLM..."
  kill ${PIDS[*]} 2>/dev/null
  # If the process is not cleaned well
  ps -u $USER -o pid,command | grep 'vllm serve' | grep -v grep | awk '{print $1}' | xargs kill
  wait
  echo "✅ Cleanup done."
}

# Trap Ctrl+C
trap 'echo ""; echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

echo "🚀 Launching vLLM model in foreground on GPU(s): $CUDA_VISIBLE_DEVICES"
CMD="python serve_vllm.py \
  --model \"$BASE_MODEL\" \
  --port $PORT"

if [ -n "$LORA_PATH" ]; then
  CMD="$CMD --lora-modules finetune=$LORA_PATH --max-lora-rank $MAX_LORA_RANK"
fi

eval $CMD

cleanup
exit 0
