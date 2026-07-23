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
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  RUN_ROOT="${OUTPUT_ROOT}/${DATASET}/${TEXT_ENCODER}"
else
  RUN_ROOT="runs/${RUN_FOLDER}"
fi
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
EARLY_STOP_LOG="${EARLY_STOP_LOG:-${RUN_ROOT}/early_stop_monitor.log}"
EARLY_STOP_STATE="${EARLY_STOP_STATE:-${RUN_ROOT}/early_stop.json}"

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
  EXTRA+=(--bert_path "${BERT_PATH}" --bert_learning_rate "${BERT_LEARNING_RATE:-3e-5}" \
    --bert_warmup_steps "${BERT_WARMUP_STEPS:-0}")
fi
if [[ "${AUTO_RESUME:-0}" == "1" ]]; then
  for checkpoint in "${RUN_ROOT}/checkpoint_dir/checkpoint.pth.tar" "${RUN_ROOT}/checkpoint_dir/model_best.pth.tar"; do
    if [[ -s "${checkpoint}" ]] && "${PYTHON_BIN}" - "${checkpoint}" >/dev/null 2>&1 <<'PY'
import sys
import torch
torch.load(sys.argv[1], map_location="cpu", weights_only=False)
PY
    then
      EXTRA+=(--resume "${checkpoint}")
      break
    fi
  done
fi

TRAIN_COMMAND=(
  "${PYTHON_BIN}" train_roma.py
  --data_name "${DATASET}" --data_root "${DATA_ROOT}" --data_path "${DATA_ROOT}"
  --vocab_path "${VOCAB_PATH}" --text_enc_type "${TEXT_ENCODER}"
  --folder_name "${RUN_FOLDER}" --output_root "${RUN_ROOT}" --module_name SGR --noise_ratio 0
  --num_epochs "${NUM_EPOCHS:-$EPOCHS}" --learning_rate "${LEARNING_RATE:-$LR}"
  --lr_update "${LR_UPDATE_OVERRIDE:-$LR_UPDATE}" --batch_size "${BATCH_SIZE:-$BATCH}"
  --workers "${WORKERS:-8}" --img_dim 1024 --embed_size 1024 --num_regions 200 --gpu "${GPU_ID}"
  "${EXTRA[@]}" "${@:3}"
)

if [[ "${EARLY_STOP_PATIENCE}" =~ ^[0-9]+$ ]] && [[ "${EARLY_STOP_PATIENCE}" -gt 0 ]]; then
  mkdir -p "$(dirname "${EARLY_STOP_LOG}")" "$(dirname "${EARLY_STOP_STATE}")"
  "${PYTHON_BIN}" "${PWD}/../../../tools/train_with_early_stop.py" \
    --patience "${EARLY_STOP_PATIENCE}" \
    --log "${EARLY_STOP_LOG}" \
    --state-json "${EARLY_STOP_STATE}" \
    -- "${TRAIN_COMMAND[@]}"
else
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${TRAIN_COMMAND[@]}"
fi
