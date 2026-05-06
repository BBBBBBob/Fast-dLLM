#!/bin/bash
# Download ALLaVA-4V LAION (JSON + image zip chunks), unzip into allava_laion/images/,
# and write source_training_env.sh for Fast-dVLM (custom_multi_modal).
# Hub layout: allava_laion/image_chunks/images_{0..9}.zip (~9GB each; full ~90GB).
#
# Usage:
#   bash fast_dvlm/data/download_example_dataset.sh allava
#   bash fast_dvlm/data/download_example_dataset.sh allava /path/to/ALLaVA-4V
#
# Only some chunks (comma-separated indices, for smoke tests / partial disk):
#   IMAGE_CHUNKS=0 bash fast_dvlm/data/download_example_dataset.sh allava
#   IMAGE_CHUNKS=0,1,2 bash ...
#
# JSON only (no image zips):
#   JSON_ONLY=1 bash fast_dvlm/data/download_example_dataset.sh allava
#
# Train (full training needs all chunks unpacked, or subset of data / filter JSON yourself):
#   source /path/to/ALLaVA-4V/source_training_env.sh
#   bash fast_dvlm/train_scripts/finetune_multimodal_example.sh

set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" != "allava" ]]; then
  echo "Usage: ${0##*/} allava [OUTPUT_DIR]"
  echo "  Default OUTPUT_DIR: ${_script_dir}/ALLaVA-4V"
  echo "  JSON_ONLY=1     — only LAION JSON"
  echo "  IMAGE_CHUNKS=0,1 — only listed chunk indices (default: 0–9 all). Each zip ~9GB."
  exit 1
fi

_out="${2:-${_script_dir}/ALLaVA-4V}"
mkdir -p "${_out}/allava_laion"
ALLAVA_ROOT="$(cd "${_out}" && pwd)"

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: need Hugging Face CLI (hf). Install: pip install -U huggingface_hub"
  exit 1
fi

echo "Downloading LAION JSON -> ${ALLAVA_ROOT}"
hf download FreedomIntelligence/ALLaVA-4V \
  --repo-type dataset \
  --local-dir "${ALLAVA_ROOT}" \
  --include 'allava_laion/*.json'

LAION="${ALLAVA_ROOT}/allava_laion"
CHUNKS_DIR="${LAION}/image_chunks"
mkdir -p "${CHUNKS_DIR}"

if [[ "${JSON_ONLY:-0}" == "1" ]]; then
  echo "JSON_ONLY=1: skipping image chunks."
else
  if [[ -n "${IMAGE_CHUNKS:-}" ]]; then
    IFS=',' read -r -a CHUNK_INDICES <<< "${IMAGE_CHUNKS// /}"
  else
    CHUNK_INDICES=(0 1 2 3 4 5 6 7 8 9)
  fi

  if ! command -v unzip >/dev/null 2>&1; then
    echo "Error: unzip not found."
    exit 1
  fi

  for i in "${CHUNK_INDICES[@]}"; do
    if [[ ! "${i}" =~ ^[0-9]+$ ]] || [[ "${i}" -lt 0 ]] || [[ "${i}" -gt 9 ]]; then
      echo "Error: invalid chunk index '${i}' (expect 0–9)."
      exit 1
    fi
    chunk_zip="${CHUNKS_DIR}/images_${i}.zip"
    marker="${LAION}/.chunk_${i}_unzipped"

    if [[ ! -f "${chunk_zip}" ]]; then
      echo "Downloading image chunk images_${i}.zip ..."
      hf download FreedomIntelligence/ALLaVA-4V \
        --repo-type dataset \
        --local-dir "${ALLAVA_ROOT}" \
        --include "allava_laion/image_chunks/images_${i}.zip"
    else
      echo "Using existing ${chunk_zip}"
    fi

    if [[ -f "${marker}" ]]; then
      echo "Chunk ${i} already extracted (${marker})."
    else
      echo "Unzipping images_${i}.zip -> ${LAION}/ ..."
      unzip -q -o "${chunk_zip}" -d "${LAION}/"
      touch "${marker}"
    fi
  done

  python3 - <<PY
import os, re
root = r"${LAION}/images"
laion = r"${LAION}"
json_path = os.path.join(laion, "ALLaVA-Instruct-LAION-4V.json")
if not os.path.isdir(root):
    print(f"Warning: missing {root}")
else:
    with os.scandir(root) as it:
        if next(it, None) is None:
            print(f"Warning: empty {root}")
        else:
            print(f"OK: {root} has files")
if os.path.isfile(json_path):
    with open(json_path, "rb") as f:
        blob = f.read(4 * 1024 * 1024)
    m = re.search(rb'"image"\s*:\s*"([^"]+)"', blob)
    if m:
        rel = m.group(1).decode("utf-8", errors="replace")
        abs_path = os.path.normpath(os.path.join(r"${ALLAVA_ROOT}", rel.replace("/", os.sep)))
        print(f"Sample JSON image field: {rel}")
        print(f"Resolved path exists: {os.path.isfile(abs_path)} ({abs_path})")
PY
fi

ENV_SH="${ALLAVA_ROOT}/source_training_env.sh"
cat > "${ENV_SH}" <<'ENVEOF'
#!/usr/bin/env bash
# Auto-generated for ALLaVA-4V LAION + Fast-dVLM (custom_multi_modal)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export DATASET_PATH="${ROOT}/allava_laion/ALLaVA-Instruct-LAION-4V.json"
export IMAGE_FOLDER="${ROOT}"
ENVEOF
chmod +x "${ENV_SH}"

echo ""
echo "Wrote ${ENV_SH}"
echo "Train:"
echo "  source ${ENV_SH}"
echo "  bash fast_dvlm/train_scripts/finetune_multimodal_example.sh"
