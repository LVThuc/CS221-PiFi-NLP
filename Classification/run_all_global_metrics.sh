#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/rerun_all preds/rerun_all cache data preprocessed_rerun models_rerun checkpoints_rerun

export HF_HOME="${HF_HOME:-$PWD/cache/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
unset TRANSFORMERS_CACHE

DATASETS_STR="${DATASETS_STR:-imdb uit_vsfc chnsenticorp zh_sentiment}"
ENCODERS_STR="${ENCODERS_STR:-bert mbert}"
LLMS_STR="${LLMS_STR:-qwen2_0.5b qwen2_1.5b qwen2_7b}"

DEFAULT_LR="${DEFAULT_LR:-5e-5}"
IMDB_MBERT_LRS_STR="${IMDB_MBERT_LRS_STR:-2e-5 5e-5}"

NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

CACHE_PATH="./cache"
PREPROCESS_PATH="./preprocessed_rerun"
DATA_PATH="./data"

LOG_DIR="./logs/rerun_all"
PRED_DIR="./preds/rerun_all"
MANIFEST="${LOG_DIR}/manifest.csv"

read -ra DATASETS <<< "$DATASETS_STR"
read -ra ENCODERS <<< "$ENCODERS_STR"
read -ra LLMS <<< "$LLMS_STR"
read -ra IMDB_MBERT_LRS <<< "$IMDB_MBERT_LRS_STR"

echo "dataset,encoder,method,llm,lr,num_epochs,batch_size,train_log,test_log,pred_file,model_path,checkpoint_path" > "$MANIFEST"

echo "=== Config ==="
echo "DATASETS      = ${DATASETS[*]}"
echo "ENCODERS      = ${ENCODERS[*]}"
echo "LLMS          = ${LLMS[*]}"
echo "DEFAULT_LR    = $DEFAULT_LR"
echo "IMDB_MBERT_LRS= ${IMDB_MBERT_LRS[*]}"
echo "NUM_EPOCHS    = $NUM_EPOCHS"
echo "BATCH_SIZE    = $BATCH_SIZE"
echo "SKIP_EXISTING = $SKIP_EXISTING"
echo "HF_HOME       = $HF_HOME"

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

safe_name () {
  echo "$1" | sed 's/\./_/g' | sed 's/-/m/g'
}

get_lrs () {
  local dataset="$1"
  local encoder="$2"

  if [[ "$dataset" == "imdb" && "$encoder" == "mbert" ]]; then
    echo "${IMDB_MBERT_LRS[*]}"
  else
    echo "$DEFAULT_LR"
  fi
}

run_preprocess () {
  local dataset="$1"
  local encoder="$2"

  local log="${LOG_DIR}/${dataset}__${encoder}__preprocess.log"

  echo
  echo "===================================================="
  echo "Preprocessing ${dataset} + ${encoder}"
  echo "===================================================="

  if [[ "$SKIP_EXISTING" == "1" && -f "$log" ]]; then
    echo "Skip existing preprocess log: $log"
    return
  fi

  python main.py \
    --task classification \
    --job=preprocessing \
    --task_dataset="$dataset" \
    --model_type="$encoder" \
    --cache_path "$CACHE_PATH" \
    --preprocess_path "$PREPROCESS_PATH" \
    --data_path "$DATA_PATH" \
    --use_wandb false 2>&1 | tee "$log"
}

run_train_test () {
  local dataset="$1"
  local encoder="$2"
  local method="$3"
  local llm="$4"
  local lr="$5"

  local llm_safe
  llm_safe=$(safe_name "$llm")
  local lr_safe
  lr_safe=$(safe_name "$lr")

  local tag="${dataset}__${encoder}__${method}__${llm_safe}__lr${lr_safe}"
  local model_path="./models_rerun/${tag}"
  local checkpoint_path="./checkpoints_rerun/${tag}"
  local train_log="${LOG_DIR}/${tag}__train.log"
  local test_log="${LOG_DIR}/${tag}__test.log"
  local pred_file="${PRED_DIR}/${tag}.csv"

  echo "$dataset,$encoder,$method,$llm,$lr,$NUM_EPOCHS,$BATCH_SIZE,$train_log,$test_log,$pred_file,$model_path,$checkpoint_path" >> "$MANIFEST"

  echo
  echo "===================================================="
  echo "Run: dataset=$dataset encoder=$encoder method=$method llm=$llm lr=$lr"
  echo "===================================================="

  if [[ "$SKIP_EXISTING" == "1" && -f "$train_log" && -f "$test_log" && -f "$pred_file" ]]; then
    echo "Skip existing run: $tag"
    return
  fi

  mkdir -p "$model_path" "$checkpoint_path"

  if [[ "$method" == "base" ]]; then
    python main.py \
      --task classification \
      --job=training \
      --task_dataset="$dataset" \
      --test_dataset="$dataset" \
      --model_type="$encoder" \
      --method=base \
      --llm_model=qwen2_0.5b \
      --layer_num=-1 \
      --padding=cls \
      --batch_size "$BATCH_SIZE" \
      --num_epochs "$NUM_EPOCHS" \
      --learning_rate "$lr" \
      --cache_path "$CACHE_PATH" \
      --preprocess_path "$PREPROCESS_PATH" \
      --data_path "$DATA_PATH" \
      --model_path "$model_path" \
      --checkpoint_path "$checkpoint_path" \
      --use_wandb false 2>&1 | tee "$train_log"

    SAVE_PRED="$pred_file" python main.py \
      --task classification \
      --job=testing \
      --task_dataset="$dataset" \
      --test_dataset="$dataset" \
      --model_type="$encoder" \
      --method=base \
      --llm_model=qwen2_0.5b \
      --layer_num=-1 \
      --padding=cls \
      --batch_size "$BATCH_SIZE" \
      --cache_path "$CACHE_PATH" \
      --preprocess_path "$PREPROCESS_PATH" \
      --data_path "$DATA_PATH" \
      --model_path "$model_path" \
      --checkpoint_path "$checkpoint_path" \
      --use_wandb false 2>&1 | tee "$test_log"

  else
    python main.py \
      --task classification \
      --job=training \
      --task_dataset="$dataset" \
      --test_dataset="$dataset" \
      --model_type="$encoder" \
      --method=pifi \
      --llm_model="$llm" \
      --layer_num=-1 \
      --padding=cls \
      --batch_size "$BATCH_SIZE" \
      --num_epochs "$NUM_EPOCHS" \
      --learning_rate "$lr" \
      --cache_path "$CACHE_PATH" \
      --preprocess_path "$PREPROCESS_PATH" \
      --data_path "$DATA_PATH" \
      --model_path "$model_path" \
      --checkpoint_path "$checkpoint_path" \
      --use_wandb false 2>&1 | tee "$train_log"

    SAVE_PRED="$pred_file" python main.py \
      --task classification \
      --job=testing \
      --task_dataset="$dataset" \
      --test_dataset="$dataset" \
      --model_type="$encoder" \
      --method=pifi \
      --llm_model="$llm" \
      --layer_num=-1 \
      --padding=cls \
      --batch_size "$BATCH_SIZE" \
      --cache_path "$CACHE_PATH" \
      --preprocess_path "$PREPROCESS_PATH" \
      --data_path "$DATA_PATH" \
      --model_path "$model_path" \
      --checkpoint_path "$checkpoint_path" \
      --use_wandb false 2>&1 | tee "$test_log"
  fi
}

for dataset in "${DATASETS[@]}"; do
  for encoder in "${ENCODERS[@]}"; do
    run_preprocess "$dataset" "$encoder"

    lrs="$(get_lrs "$dataset" "$encoder")"
    read -ra LR_LIST <<< "$lrs"

    for lr in "${LR_LIST[@]}"; do
      run_train_test "$dataset" "$encoder" "base" "-" "$lr"

      for llm in "${LLMS[@]}"; do
        run_train_test "$dataset" "$encoder" "pifi" "$llm" "$lr"
      done
    done
  done
done

echo
echo "All runs finished. Manifest: $MANIFEST"
