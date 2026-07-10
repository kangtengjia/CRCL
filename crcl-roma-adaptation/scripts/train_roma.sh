#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: train_roma.sh <scenedepict|scanrefer|nr3d|3dllm> <bigru|bert>}"
TEXT_ENCODER="${2:?usage: train_roma.sh <dataset> <bigru|bert>}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/ktj/miniconda3/envs/oneformer3d/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/data}"
VOCAB_PATH="${VOCAB_PATH:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/vocab}"
BERT_PATH="${BERT_PATH:-/home/ktj/Projects/RoMa/pretrained/bert-base-uncased}"
RUN_FOLDER="${RUN_FOLDER:-roma/${DATASET}/${TEXT_ENCODER}}"
RUN_ROOT="runs/${RUN_FOLDER}"

cd "$(dirname "$0")/.."
read -r EPOCHS LR_UPDATE LR BATCH <<EOF
$(${PYTHON_BIN} - "${DATASET}" "${TEXT_ENCODER}" <<'PY'
import sys
from roma_runs import RUN_PRESETS
preset = RUN_PRESETS[(sys.argv[1], sys.argv[2])]
print(preset.epochs, preset.lr_update, preset.learning_rate, preset.batch_size)
PY
)
EOF

EXTRA=()
if [[ "${TEXT_ENCODER}" == "bert" ]]; then
  EXTRA+=(--bert_path "${BERT_PATH}")
fi
if [[ "${AUTO_RESUME:-0}" == "1" && -s "${RUN_ROOT}/checkpoint_dir/checkpoint.pth.tar" ]]; then
  EXTRA+=(--resume "${RUN_ROOT}/checkpoint_dir/checkpoint.pth.tar")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_roma.py \
  --data_name "${DATASET}" --data_root "${DATA_ROOT}" --data_path "${DATA_ROOT}" \
  --vocab_path "${VOCAB_PATH}" --text_enc_type "${TEXT_ENCODER}" \
  --folder_name "${RUN_FOLDER}" --module_name SGR --noise_ratio 0 \
  --num_epochs "${NUM_EPOCHS:-$EPOCHS}" --learning_rate "${LEARNING_RATE:-$LR}" \
  --lr_update "${LR_UPDATE_OVERRIDE:-$LR_UPDATE}" --batch_size "${BATCH_SIZE:-$BATCH}" \
  --workers "${WORKERS:-8}" --img_dim 1024 --embed_size 1024 --num_regions 200 --gpu "${GPU_ID}" \
  "${EXTRA[@]}" "${@:3}"
