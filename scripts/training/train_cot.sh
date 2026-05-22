set -e
set -x

MODEL=$1
DATASETS=(
    "agent-distillation/Qwen2.5-32B-Instruct_cot_trajectories_2k"
)
SAVENAME="qwen2.5_32B_teacher"
JOINED_DATASETS=$(IFS=" "; echo "${DATASETS[*]}")

sh exps_research/scripts_train/finetune_sft_cot.sh \
    $MODEL \
    "$JOINED_DATASETS" \
    $SAVENAME
