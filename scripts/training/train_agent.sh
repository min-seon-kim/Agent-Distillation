set -e
set -x

MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
EPOCH=${2:-2}
DATASETS=(
  "agent-distillation/Qwen2.5-32B-Instruct_agent_trajectories_2k"
)
SAVENAME="qwen2.5_32B_teacher"
JOINED_DATASETS=$(IFS=" "; echo "${DATASETS[*]}")

sh exps_research/scripts_train/finetune_sft_agent.sh \
    $MODEL \
    "$JOINED_DATASETS" \
    $SAVENAME \
    $EPOCH
