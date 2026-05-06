#!/bin/bash
# Example Fast-dVLM finetuning (multimodal MDM). Mirrors v2/train_scripts/finetune_alpaca.sh style.
# Uses only files inside this Fast-dLLM repo: vendored lmflow under third_party/ and
# fast_dvlm/train_scripts/finetune_dvlm.py.
#
# Prerequisites: pip install -e v2/  (or set PYTHONPATH to Fast-dLLM/third_party), torch, deepspeed,
#   transformers, Pillow, datasets (see LMFlow multimodal / custom_multi_modal docs).
#
# Dataset (LMFlow custom_multi_modal): DATASET_PATH is one JSON file (list of samples) with
# ``image`` + ``conversations`` (LLaVA-style). IMAGE_FOLDER is the directory containing image files.
# See CustomMultiModalDataset in third_party/lmflow/datasets/multi_modal_dataset.py.
#
# Data (ALLaVA-4V): https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V
#   bash fast_dvlm/data/download_example_dataset.sh allava   # JSON + image_chunks zips + unzip + source_training_env.sh
#   source .../fast_dvlm/data/ALLaVA-4V/source_training_env.sh
#   IMAGE_CHUNKS=0 downloads only shard 0 for a smoke run; default 0–9 is full (~90 GB). JSON_ONLY=1 fetches JSON only.

set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo_root="$(cd "${_script_dir}/../.." && pwd)"
export PYTHONPATH="${_repo_root}/third_party:${PYTHONPATH:-}"

# Default ALLaVA-4V layout (same as fast_dvlm/data/ALLaVA-4V/source_training_env.sh).
_default_allava="${_repo_root}/fast_dvlm/data/ALLaVA-4V"
if [[ -z "${DATASET_PATH:-}" ]]; then
  export DATASET_PATH="${_default_allava}/allava_laion/ALLaVA-Instruct-LAION-4V.json"
fi
if [[ -z "${IMAGE_FOLDER:-}" ]]; then
  export IMAGE_FOLDER="${_default_allava}"
fi

output_dir="${OUTPUT_DIR:-${_repo_root}/output_models/finetune_fast_dVLM_3B_example}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Error: dataset JSON not found: ${DATASET_PATH}"
  echo "  Download ALLaVA-4V (see script header) or set DATASET_PATH."
  exit 1
fi
if [[ ! -d "${IMAGE_FOLDER}" ]]; then
  echo "Error: image root directory not found: ${IMAGE_FOLDER}"
  echo "  Set IMAGE_FOLDER to the directory that contains paths like allava_laion/images/...."
  exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

if [[ -n "${DEEPSPEED_ARGS:-}" ]]; then
  deepspeed_args="${DEEPSPEED_ARGS}"
else
  deepspeed_args="--master_port=${MASTER_PORT:-11001}"
fi

latest_checkpoint=""
if [[ -d "${output_dir}" ]]; then
  latest_checkpoint="$(find "${output_dir}" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | sort -V | tail -1 || true)"
  [[ -n "${latest_checkpoint}" ]] && echo "Found latest checkpoint: ${latest_checkpoint}"
fi
resume_arg=""
if [[ -n "${latest_checkpoint}" ]]; then
  resume_arg="--resume_from_checkpoint $(printf '%q' "${latest_checkpoint}")"
fi

ds_config="${DEEPSPEED_CONFIG:-${_repo_root}/v2/configs/ds_config_zero2_no_offload.json}"
if [[ ! -f "${ds_config}" ]]; then
  echo "Error: DeepSpeed config not found: ${ds_config}"
  exit 1
fi

cd "${_repo_root}"

cmd="deepspeed ${deepspeed_args} \
  fast_dvlm/train_scripts/finetune_dvlm.py \
    --model_name_or_path $(printf '%q' "${MODEL_PATH:-/home/chengyuew/chengyuew/workspace/LMFlow/output_models/Fast_dVLM_3B}") \
    --tokenizer_name $(printf '%q' "${TOKENIZER_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}") \
    --trust_remote_code ${TRUST_REMOTE_CODE:-1} \
    --dataset_path $(printf '%q' "${DATASET_PATH}") \
    --image_folder $(printf '%q' "${IMAGE_FOLDER}") \
    --output_dir $(printf '%q' "${output_dir}") \
    ${resume_arg} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS:-1} \
    --learning_rate ${LEARNING_RATE:-2e-5} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE:-constant_with_warmup} \
    --warmup_ratio ${WARMUP_RATIO:-0.03} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE:-1} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS:-1} \
    --deepspeed $(printf '%q' "${ds_config}") \
    --bf16 \
    --run_name ${RUN_NAME:-finetune_fast_dvlm} \
    --validation_split_percentage ${VALIDATION_SPLIT_PERCENTAGE:-0} \
    --logging_steps ${LOGGING_STEPS:-1} \
    --do_train \
    --ddp_timeout ${DDP_TIMEOUT:-72000} \
    --save_steps ${SAVE_STEPS:-1000} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS:-8} \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS:-32} \
    --save_total_limit ${SAVE_TOTAL_LIMIT:-10} \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING:-1} \
    --report_to ${REPORT_TO:-none} \
    ${MAX_STEPS:+--max_steps ${MAX_STEPS}}"

echo "${cmd}"
eval "${cmd}"
