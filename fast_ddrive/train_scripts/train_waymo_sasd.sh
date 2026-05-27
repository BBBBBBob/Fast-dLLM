#!/bin/bash
# Fast-dDrive Waymo-E2E SASD finetuning launcher.
#
# Mirrors fast_dvlm/train_scripts/finetune_multimodal_example.sh: a single
# DeepSpeed launcher that forwards to finetune_fast_ddrive.py with the
# canonical paper recipe (MDM + deep JSON scaffold + Section-Importance-
# Weighted Loss + Section-Adaptive Noise Schedule).
#
# Required env:
#   DATASET_PATH   — Waymo training JSON (see data/README.md for the schema)
#   IMAGE_FOLDER   — root that ``DATASET_PATH``'s image paths are relative to
#
# Optional env (all have paper-canonical defaults):
#   MODEL_PATH                  default: Qwen/Qwen2.5-VL-3B-Instruct (base)
#   OUTPUT_DIR                  default: <repo>/output_models/finetune_fast_ddrive
#   NUM_TRAIN_EPOCHS            default: 2
#   LEARNING_RATE               default: 1e-5
#   PER_DEVICE_TRAIN_BATCH_SIZE default: 1
#   GRADIENT_ACCUMULATION_STEPS default: 4
#   MAX_STEPS                   default: unset (use epochs)
#
# Usage:
#   DATASET_PATH=/path/to/waymo_train.json IMAGE_FOLDER=/path/to/images \
#     bash fast_ddrive/train_scripts/train_waymo_sasd.sh

set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo_root="$(cd "${_script_dir}/../.." && pwd)"
export PYTHONPATH="${_repo_root}/third_party:${PYTHONPATH:-}"

: "${DATASET_PATH:?set DATASET_PATH to the Waymo training JSON (see data/README.md)}"
: "${IMAGE_FOLDER:?set IMAGE_FOLDER to the directory containing the images referenced by DATASET_PATH}"

output_dir="${OUTPUT_DIR:-${_repo_root}/output_models/finetune_fast_ddrive}"
mkdir -p "${output_dir}"

if [[ -n "${DEEPSPEED_ARGS:-}" ]]; then
  deepspeed_args="${DEEPSPEED_ARGS}"
else
  deepspeed_args="--master_port=${MASTER_PORT:-11001}"
fi

ds_config="${DEEPSPEED_CONFIG:-${_repo_root}/v2/configs/ds_config_zero2_no_offload.json}"
if [[ ! -f "${ds_config}" ]]; then
  echo "Error: DeepSpeed config not found: ${ds_config}"
  exit 1
fi

# SASD section-weighted loss + per-section Beta noise schedule (paper canonical).
section_loss_weights='{"critical_objects":1.5,"explanation":1.0,"future_meta_behavior":2.0,"trajectory":3.0}'
section_noise_schedule='{"critical_objects":"1.0,2.0","explanation":"1.0,1.0","future_meta_behavior":"1.0,1.5","trajectory":"2.0,1.0"}'

cd "${_repo_root}"

deepspeed ${deepspeed_args} \
  fast_ddrive/train_scripts/finetune_fast_ddrive.py \
    --model_name_or_path "${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}" \
    --tokenizer_name "${TOKENIZER_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}" \
    --conversation_template qwen2_5_no_reasoning \
    --trust_remote_code 1 \
    --dataset_path "${DATASET_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --output_dir "${output_dir}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-2}" \
    --learning_rate "${LEARNING_RATE:-1e-5}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE:-constant_with_warmup}" \
    --warmup_ratio "${WARMUP_RATIO:-0.03}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --deepspeed "${ds_config}" \
    --bf16 \
    --do_train \
    --gradient_checkpointing \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --save_steps "${SAVE_STEPS:-1000}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
    --validation_split_percentage 0 \
    --report_to "${REPORT_TO:-none}" \
    --mdm 1 \
    --bd_size 32 \
    --block_size 2048 \
    --learn_padding 1 \
    --complementary_mask 1 \
    --always_mask_im_end 1 \
    --use_block_causal_mask 1 \
    --deep_json_scaffold 1 \
    --section_loss_weights "${section_loss_weights}" \
    --section_noise_schedule "${section_noise_schedule}" \
    ${MAX_STEPS:+--max_steps ${MAX_STEPS}}
