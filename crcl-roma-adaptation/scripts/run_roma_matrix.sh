#!/usr/bin/env bash
set -euo pipefail

DATASETS="${DATASETS:-scenedepict,scanrefer,nr3d,3dllm}"
TEXT_ENCODERS="${TEXT_ENCODERS:-bigru,bert}"
IFS=',' read -ra DATASET_LIST <<<"${DATASETS}"
IFS=',' read -ra TEXT_ENCODER_LIST <<<"${TEXT_ENCODERS}"
for dataset in "${DATASET_LIST[@]}"; do
  for text_encoder in "${TEXT_ENCODER_LIST[@]}"; do
    bash "$(dirname "$0")/train_roma.sh" "${dataset}" "${text_encoder}"
  done
done
