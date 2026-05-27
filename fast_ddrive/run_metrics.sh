#!/usr/bin/env bash
# Fast-dDrive — official Waymo E2E ADE / RFS scoring launcher.
#
# Consumes the predictions.json written by `run_eval.sh` (i.e.
# `eval/batch_inference.py`) and computes ADE@3s, ADE@5s, and the Rater
# Feedback Score (RFS) against the Waymo Open Dataset ground truth.
#
# This step depends on `tensorflow` + `waymo_open_dataset`, which conflict
# with the inference stack — install them in a separate env (we use one
# called `autovla`).  See `data/README.md` for setup details.
#
# Required env:
#   PRED_JSON  — predictions.json produced by run_eval.sh
#   GT         — either a TFRecord glob (e.g. '/path/to/val*.tfrecord*')
#                or a pre-computed gt_dict pickle (.pkl) from an earlier run
#
# Optional env:
#   OUTPUT_DIR — default: <dirname of PRED_JSON>/waymo_metrics
#   PYTHON     — interpreter to use (default: python3 on $PATH).  Point this
#                at the autovla env's python when running on a clean shell.

set -eo pipefail

FAST_DDRIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${PRED_JSON:?Set PRED_JSON to the predictions.json written by run_eval.sh.}"
: "${GT:?Set GT to a Waymo TFRecord glob or a gt_dict pickle (.pkl).}"

OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${PRED_JSON}")/waymo_metrics}"
PYTHON="${PYTHON:-python3}"

echo "=========================================="
echo "Fast-dDrive Waymo metrics"
echo "  PRED_JSON: ${PRED_JSON}"
echo "  GT:        ${GT}"
echo "  OUTPUT:    ${OUTPUT_DIR}"
echo "=========================================="

"${PYTHON}" "${FAST_DDRIVE_ROOT}/eval/evaluate_waymo_metrics.py" \
    --pred_json "${PRED_JSON}" \
    --gt "${GT}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
