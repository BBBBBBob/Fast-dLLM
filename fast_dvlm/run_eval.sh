#!/usr/bin/env bash
# VLMEvalKit eval (default TASK=DocVQA_VAL). Inference uses checkpoint ``modeling.py`` ``generate`` (trust_remote_code).

set -eo pipefail
FAST_DVLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${FAST_DVLM_ROOT}/.." && pwd)"
VLM_DIR="${VLM_DIR:-${REPO_ROOT}/third_party/VLMEvalKit}"
VLMEVAL_PY="${FAST_DVLM_ROOT}/vlmeval_run.py"
TASK="${TASK:-DocVQA_VAL}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

show_help() {
    echo "VLMEval eval (default TASK=DocVQA_VAL)"
    echo "  pip install -e ${REPO_ROOT}/third_party/VLMEvalKit"
    echo "  bash ${FAST_DVLM_ROOT}/run_eval.sh [--help|-h]"
    echo ""
    echo "Required: MODEL_PATH — same as run_chatbot.py --model-name:"
    echo "         HuggingFace id (default chatbot: Efficient-Large-Model/Fast_dVLM_3B) or local checkpoint dir."
    echo "Optional: TASK OUTPUT_BASE NUM_GPUS VLM_DIR"
    echo "          PROCESSOR_PATH (default: same as MODEL_PATH; set if weights dir has no processor)"
    echo "          DATASET_CLASS (default ImageVQADataset) MAX_TOKENS BLOCK_SIZE MASK_TOKEN TORCH_DTYPE"
    exit 0
}
[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && show_help

if [ ! -f "${VLM_DIR}/run.py" ]; then
    echo "ERROR: VLMEvalKit not found at: ${VLM_DIR}"
    exit 1
fi
if [ ! -f "${VLMEVAL_PY}" ]; then
    echo "ERROR: Missing ${VLMEVAL_PY}"
    exit 1
fi
if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: Set MODEL_PATH (see --help)."
    exit 1
fi

if [ -d "${MODEL_PATH}" ]; then
    MODEL_PATH_ABS=$(cd "${MODEL_PATH}" && pwd)
elif [ -d "${REPO_ROOT}/${MODEL_PATH}" ]; then
    MODEL_PATH_ABS=$(cd "${REPO_ROOT}/${MODEL_PATH}" && pwd)
else
    # HuggingFace Hub id or other string passed to from_pretrained (same as chatbot --model-name)
    MODEL_PATH_ABS="${MODEL_PATH}"
fi

if [ -z "${OUTPUT_BASE:-}" ]; then
    if [ -d "${MODEL_PATH_ABS}" ]; then
        CKPT_NAME=$(basename "${MODEL_PATH_ABS}")
        if [[ "${CKPT_NAME}" == checkpoint-* ]]; then
            OUTPUT_BASE="$(dirname "${MODEL_PATH_ABS}")/eval/${CKPT_NAME}"
        else
            OUTPUT_BASE="${MODEL_PATH_ABS}/eval"
        fi
    else
        SAFE=$(echo "${MODEL_PATH_ABS}" | tr '/' '_')
        OUTPUT_BASE="${REPO_ROOT}/eval_outputs/vlmeval_${SAFE}"
    fi
fi
mkdir -p "${OUTPUT_BASE}"
OUTPUT_BASE="$(cd "${OUTPUT_BASE}" && pwd)"
TASK_OUTPUT="${OUTPUT_BASE}/${TASK}"
mkdir -p "${TASK_OUTPUT}"

if [ -z "${NUM_GPUS}" ]; then
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d '[:space:]' || echo 1)
    if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [ "${NUM_GPUS}" -eq 0 ]; then
        NUM_GPUS=1
    fi
fi

export CFG_PATH="${TASK_OUTPUT}/config.json"
export MODEL_PATH_ABS="${MODEL_PATH_ABS}"
export DATASETS="${TASK}"
[ -n "${PROCESSOR_PATH:-}" ] && export PROCESSOR_PATH
export DATASET_CLASS="${DATASET_CLASS:-ImageVQADataset}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export BLOCK_SIZE="${BLOCK_SIZE:-}"
export MASK_TOKEN="${MASK_TOKEN:-|<MASK>|}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

echo "Writing config -> ${CFG_PATH}"
python3 "${VLMEVAL_PY}" write-config

cd "${VLM_DIR}"
MASTER_PORT=${MASTER_PORT:-$(python3 -c "import random; print(random.randint(20000,65000))")}
echo "Running ${TASK} -> ${TASK_OUTPUT}"
torchrun --master_port="${MASTER_PORT}" --nproc-per-node="${NUM_GPUS}" "${VLMEVAL_PY}" \
    --config "${CFG_PATH}" --work-dir "${TASK_OUTPUT}" --mode all --verbose 2>&1 | tee "${TASK_OUTPUT}/eval.log"

echo "Done. OUTPUT=${TASK_OUTPUT}"
