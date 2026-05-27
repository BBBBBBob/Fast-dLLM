# Waymo End-to-End Dataset Layout (Fast-dDrive)

Fast-dDrive trains and evaluates on the Waymo End-to-End Driving (WOD-E2E)
dataset. We do **not** redistribute Waymo data — follow the official process and
preprocess it into the JSON layout below.

## 1. Obtain the raw data

1. Apply for access to Waymo Open Dataset: https://waymo.com/open/
2. Download the WOD-E2E `train` and `val` tfrecord shards.
3. Optional: pre-compute the GT pickle for evaluation (see below).

## 2. Convert to Fast-dDrive JSON

Each training / val sample is a JSON object roughly of the form:

```json
{
  "id": "scene_uuid",
  "image": [
    "waymo_train/scene_uuid/frame_0.jpg",
    "waymo_train/scene_uuid/frame_1.jpg",
    "..."
  ],
  "conversations": [
    {"from": "human", "value": "<image>...<image>Drive task prompt..."},
    {"from": "gpt",   "value": "{\"critical_objects\": {...}, \"future_meta_behavior\": {...}, \"trajectory\": \"[[+003.30,-000.01], ...]\", \"explanation\": \"...\"}"}
  ]
}
```

Notes:

- The model output is a JSON-shaped string; trajectory waypoints use the
  `+XXX.XX,-XXX.XX` zero-padded format (parsed by `eval/batch_inference.py`).


A reference preprocessing script will be added under `fast_ddrive/data/`. For
now, point the training scripts at your own conversion via env vars:

```bash
export DATASET_JSON=/path/to/waymo_train.json
export IMAGE_FOLDER=/path/to/image/root      # JSON image paths are relative to this
export EVAL_JSON=/path/to/waymo_val.json
export IMAGE_ROOT=/path/to/image/root        # same root used for eval
```

## 3. Official Waymo ADE / RFS evaluation

`eval/evaluate_waymo_metrics.py` computes ADE@3s, ADE@5s and Rater Feedback
Score (RFS). It requires `tensorflow` + `waymo-open-dataset-tf-2-12-0`, which
conflict with the inference stack — install in a separate conda env (we call
ours `autovla`):

```bash
conda create -n autovla python=3.10
conda activate autovla
pip install tensorflow==2.12.0 waymo-open-dataset-tf-2-12-0 numpy tqdm
python eval/evaluate_waymo_metrics.py \
    --pred_json /path/to/predictions.json \
    --gt_tfrecords "/path/to/val*.tfrecord*" \
    --output_dir results/
```

For faster eval on compute nodes, cache the GT dict once:

```bash
python eval/evaluate_waymo_metrics.py \
    --pred_json /path/to/predictions.json \
    --save_gt_dict_pkl gt_dict_val.pkl \
    --gt_tfrecords "/path/to/val*.tfrecord*"
# subsequent runs:
python eval/evaluate_waymo_metrics.py \
    --pred_json /path/to/predictions.json \
    --gt_dict_pkl gt_dict_val.pkl
```
