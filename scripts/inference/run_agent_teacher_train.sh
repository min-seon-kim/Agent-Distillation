#!/bin/bash
# ============================================================================ #
#  Teacher(32B) agent 궤적 생성 — 우리 환경(A40 x8, conda 없음, venv, RAM 1TB)에 맞춤
#  - math(math, math2) + factual(hotpotqa) train split 에 agent 궤적 생성
#  - --do_filtering 으로 정답필터(filtered_data/*_filtered.jsonl)까지 저장
#  - 산출물: logs/qa_results/vllm/Qwen_Qwen3-32B/<dataset>_train/...steps=5.jsonl
#  사용: bash scripts/inference/run_agent_teacher_train.sh
# ============================================================================ #

# ===================== user setting ===================== #
BASE_MODEL="Qwen/Qwen3-32B"          # 기존 math 궤적과 동일 teacher (불일치 방지)
EXP_TYPE="agent"
PORT_BASE=8000                        # vLLM (OpenAI 호환) 포트  (client: --use_single_endpoint → 8000)
MAX_TOKENS=2048

# ── 우리 환경: conda 대신 repo venv ──
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO_ROOT/keys/.venv"          # vllm 0.9.1 / faiss 1.14.1 / smolagents

# ── GPU 배정 (사용 중: 0,1,2,3,6 → 회피 / 완전 여유: 4,5,7) ──
VLLM_GPU_DEVICES="4,5"               # Qwen3-32B (TP=2; 64개 head 라 TP 는 2 또는 4)
VLLM_TP=2
VLLM_GPU_UTIL=0.92
RETRIEVER_GPU_DEVICES="7"            # encoder(e5)용 1장. faiss 인덱스는 CPU(RAM) 로 → OOM 회피
FAISS_GPU=0                           # 0=인덱스 CPU(RAM 1TB 충분), 1=GPU(샤딩, 64GB 필요)

RETRIEVER_LOG="$REPO_ROOT/retriever_server.log"   # 서버 포트 8005 (client default_tools.py 와 일치)
VLLM_LOG="$REPO_ROOT/vllm.log"

declare -A DATASETS=(
  ["hotpotqa"]="data_processor/qa_dataset/train/hotpotqa_1000_20250402.json"   # factual (retriever 필요)
  ["math"]="data_processor/math_dataset/train/math_1000_20250414.json"
  ["math2"]="data_processor/math_dataset/train/math_medium_1000_20250430.json"
)
declare -A PREFIXS=(
  ["hotpotqa"]="logs/qa_results/vllm/Qwen_Qwen3-32B/hotpotqa_1000_20250402_train/prefix_memory/Qwen3-32B_temp=0.0_seed=42_type=reasoning.json"
  ["math"]="logs/qa_results/vllm/Qwen_Qwen3-32B/math_1000_20250414_train/prefix_memory/Qwen3-32B_temp=0.0_seed=42_type=reasoning.json"
  ["math2"]="logs/qa_results/vllm/Qwen_Qwen3-32B/math_medium_1000_20250430_train/prefix_memory/Qwen3-32B_temp=0.0_seed=42_type=reasoning.json"
)
# ===================================================== #

SKIP_SERVING=false
USE_WEB_SEARCH=false
USE_PREFIX=false
for arg in "$@"; do
  case $arg in
    --skip-serving)   SKIP_SERVING=true ;;
    --use-web-search) USE_WEB_SEARCH=true ;;
    --use-prefix)     USE_PREFIX=true ;;
  esac
done

cd "$REPO_ROOT" || exit 1

# ── venv 활성화 (conda 없음) ──
if [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
  echo "🐍 venv: $VENV ($(python --version 2>&1))"
else
  echo "❌ venv not found: $VENV"; exit 1
fi

# ── 이 환경엔 ps/pgrep/pkill 가 없어 /proc 스캔으로 프로세스를 죽인다 ──
kill_matching() {  # $1: cmdline 에 포함된 패턴
  local pat="$1" d cl
  for d in /proc/[0-9]*; do
    cl=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$cl" in *"$pat"*) kill "${d#/proc/}" 2>/dev/null ;; esac
  done
}

PIDS=()
cleanup() {
  echo ""; echo "🧹 Cleaning up servers..."
  kill "${PIDS[@]}" 2>/dev/null
  kill_matching "vllm serve $BASE_MODEL"      # serve 자식까지
  kill_matching "search/retriever_server.py"
  wait 2>/dev/null
  echo "✅ All servers stopped."
}
trap 'echo ""; echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM

# ===================================================== #
# Start Retriever + vLLM servers
# ===================================================== #
if [ "$SKIP_SERVING" = false ]; then
  # ── retriever: 포트 8005 단일 인스턴스 → 기존(stale) 것이 있으면 정리 ──
  if [ "$USE_WEB_SEARCH" = false ]; then
    kill_matching "search/retriever_server.py"; sleep 2
    echo "🔍 Launching retriever (encoder GPU $RETRIEVER_GPU_DEVICES, faiss_gpu=$FAISS_GPU, port 8005)…"
    CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_DEVICES \
      python search/retriever_server.py --faiss_gpu $FAISS_GPU > "$RETRIEVER_LOG" 2>&1 &
    RETRIEVER_PID=$!
    PIDS+=($RETRIEVER_PID)
    echo "🛰️  retriever PID $RETRIEVER_PID — CPU 인덱스 로드(64GB)에 수 분 소요…"
    until grep -qE "Uvicorn running|Application startup complete" "$RETRIEVER_LOG" 2>/dev/null; do
      sleep 5
      kill -0 $RETRIEVER_PID 2>/dev/null || { echo "❌ retriever died:"; tail -n 30 "$RETRIEVER_LOG"; cleanup; exit 1; }
    done
    echo "✅ retriever ready."
  fi

  # ── vLLM: serve_vllm.py(os.system) 대신 vllm 직접 실행 → PID 정확히 관리 ──
  echo "📺 Launching vLLM (GPUs $VLLM_GPU_DEVICES, TP=$VLLM_TP, port $PORT_BASE)…"
  CUDA_VISIBLE_DEVICES=$VLLM_GPU_DEVICES \
    vllm serve "$BASE_MODEL" \
      --host 0.0.0.0 --port $PORT_BASE \
      --tensor-parallel-size $VLLM_TP \
      --dtype bfloat16 \
      --gpu-memory-utilization $VLLM_GPU_UTIL \
      --trust-remote-code \
      --reasoning-parser qwen3 > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  PIDS+=($VLLM_PID)
  echo "⏳ waiting for vLLM startup…"
  until grep -q "Application startup complete." "$VLLM_LOG" 2>/dev/null; do
    sleep 5
    kill -0 $VLLM_PID 2>/dev/null || { echo "❌ vLLM died:"; tail -n 40 "$VLLM_LOG"; cleanup; exit 1; }
  done
  echo "✅ vLLM ready."
fi

# ===================================================== #
# Run experiments (agent 궤적 생성 + 정답필터)
# ===================================================== #
RUN_EXIT_CODE=0
for dataset in "${!DATASETS[@]}"; do
  echo "🧠 Generating teacher trajectories on '$dataset' …"
  AGENT_CMD="python -m exps_research.unified_framework.run_experiment \
    --experiment_type \"$EXP_TYPE\" \
    --data_path \"${DATASETS[$dataset]}\" \
    --model_type vllm \
    --model_id \"$BASE_MODEL\" \
    --max_tokens $MAX_TOKENS \
    --n 1 --temperature 0.0 --top_p 0.8 \
    --seed 42 \
    --verbose \
    --do_filtering"

  if [ "$USE_WEB_SEARCH" = true ]; then
    AGENT_CMD="$AGENT_CMD --search_engine_type duckduckgo"
  else
    AGENT_CMD="$AGENT_CMD --search_engine_type wikipedia --multithreading --use_process_pool --use_single_endpoint"
  fi
  if [ "$USE_PREFIX" = true ]; then
    AGENT_CMD="$AGENT_CMD --prefix_memory \"${PREFIXS[$dataset]}\""
  fi

  eval $AGENT_CMD || RUN_EXIT_CODE=$?
done

cleanup

if [ $RUN_EXIT_CODE -ne 0 ]; then
  echo "⚠️ Agent script failed with exit code $RUN_EXIT_CODE"; exit $RUN_EXIT_CODE
else
  echo "✅ Script completed successfully"; exit 0
fi
