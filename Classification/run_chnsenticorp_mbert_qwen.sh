#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs cache preprocessed data models checkpoints

export HF_HOME="${HF_HOME:-$PWD/cache/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
unset TRANSFORMERS_CACHE

DATASET="chnsenticorp"
ENCODER="mbert"
BATCH_SIZE="${BATCH_SIZE:-32}"
LLMS=("qwen2_0.5b" "qwen2_1.5b" "qwen2_7b")

echo "=== Config ==="
echo "DATASET    = $DATASET"
echo "ENCODER    = $ENCODER"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "HF_HOME    = $HF_HOME"

echo
echo "=== CUDA check ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

echo
echo "=== Ensure preprocessing exists ==="
python main.py \
  --task classification \
  --job=preprocessing \
  --task_dataset="${DATASET}" \
  --model_type="${ENCODER}" \
  --cache_path ./cache \
  --preprocess_path ./preprocessed \
  --data_path ./data \
  --use_wandb false 2>&1 | tee "logs/${DATASET}_${ENCODER}_preprocess.log"

echo
echo "===================================================="
echo "Training/testing ${DATASET} + ${ENCODER} BASE"
echo "===================================================="

python main.py \
  --task classification \
  --job=training \
  --task_dataset="${DATASET}" \
  --test_dataset="${DATASET}" \
  --model_type="${ENCODER}" \
  --method=base \
  --padding=cls \
  --batch_size="${BATCH_SIZE}" \
  --cache_path ./cache \
  --num_epochs=5 \
  --preprocess_path ./preprocessed \
  --data_path ./data \
  --model_path ./models \
  --checkpoint_path ./checkpoints \
  --use_wandb false 2>&1 | tee "logs/${DATASET}_${ENCODER}_base_train.log"

python main.py \
  --task classification \
  --job=testing \
  --task_dataset="${DATASET}" \
  --test_dataset="${DATASET}" \
  --model_type="${ENCODER}" \
  --method=base \
  --padding=cls \
  --batch_size="${BATCH_SIZE}" \
  --cache_path ./cache \
  --preprocess_path ./preprocessed \
  --data_path ./data \
  --model_path ./models \
  --checkpoint_path ./checkpoints \
  --use_wandb false 2>&1 | tee "logs/${DATASET}_${ENCODER}_base_test.log"

run_one () {
  local LLM="$1"
  local SAFE_LLM
  SAFE_LLM=$(echo "$LLM" | sed 's/\./_/g')

  echo
  echo "===================================================="
  echo "Training/testing ${DATASET} + ${ENCODER} + PiFi ${LLM}"
  echo "===================================================="

  python main.py \
    --task classification \
    --job=training \
    --task_dataset="${DATASET}" \
    --test_dataset="${DATASET}" \
    --model_type="${ENCODER}" \
    --method=pifi \
    --llm_model="${LLM}" \
    --layer_num=-1 \
    --padding=cls \
    --batch_size="${BATCH_SIZE}" \
    --num_epochs=5 \
    --cache_path ./cache \
    --preprocess_path ./preprocessed \
    --data_path ./data \
    --model_path ./models \
    --checkpoint_path ./checkpoints \
    --use_wandb false 2>&1 | tee "logs/${DATASET}_${ENCODER}_pifi_${SAFE_LLM}_train.log"

  python main.py \
    --task classification \
    --job=testing \
    --task_dataset="${DATASET}" \
    --test_dataset="${DATASET}" \
    --model_type="${ENCODER}" \
    --method=pifi \
    --llm_model="${LLM}" \
    --layer_num=-1 \
    --padding=cls \
    --batch_size="${BATCH_SIZE}" \
    --cache_path ./cache \
    --preprocess_path ./preprocessed \
    --data_path ./data \
    --model_path ./models \
    --checkpoint_path ./checkpoints \
    --use_wandb false 2>&1 | tee "logs/${DATASET}_${ENCODER}_pifi_${SAFE_LLM}_test.log"
}

for LLM in "${LLMS[@]}"; do
  run_one "$LLM"
done

echo
echo "=== Summary ==="
grep -iE "Done|TEST|Best valid|accuracy|Acc|F1" logs/${DATASET}_${ENCODER}_*.log || true
