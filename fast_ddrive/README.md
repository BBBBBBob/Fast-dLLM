# Fast-dDrive: Section-Aware Diffusion VLM for End-to-End Driving

[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/xiwenyoumu/Fast-dDrive)

Fast-dDrive is a Qwen2.5-VL-based block-diffusion model for closed-loop
driving on the Waymo End-to-End Driving benchmark (WOD-E2E). It combines:

- **Section Diffusion (SD)** — iterative MDM denoising over a pre-filled JSON
  scaffold, section by section in causal order (`critical_objects` →
  `explanation` → `future_meta_behavior` → `trajectory`), with bidirectional
  attention within each section-aligned block.
- **Scaffold Spec (SS)** — scaffold-aware self-speculative decoding (MDM
  block draft + AR causal verify per block). Paper canonical, fastest
  single-rollout decoder.
- **Inference scaling (SS multi-traj)** — shared deterministic SS prefix for
  sections 1–3, N rollouts on the trajectory section, equal-weight averaging.
- **SASD training** — Section-Importance-Weighted Loss + Section-Adaptive
  (Beta) Noise Schedule. Per-section weights (trajectory=3.0, fmb=2.0,
  critical_objects=1.5, explanation=1.0) at zero inference overhead.

Paper headline results on WOD-E2E test set (Qwen2.5-VL-3B, single H100):

| Mode                          | RFS ↑ | ADE@3s ↓ | ADE@5s ↓ | TPS ↑  | Tok/Step ↑ |
|------------------------------:|:-----:|:--------:|:--------:|:------:|:----------:|
| **Scaffold Spec**             | 7.823 | 1.254    | 2.907    | 210.4  | 4.90       |
| + Inference scaling (N=4)     | 7.827 | 1.240    | 2.821    | 114.7  | 2.76       |

On the WOD-E2E val set, Scaffold Spec runs at 1919 ms / sample (4.1× over
the AR baseline); fused with SGLang the same configuration drops to
665 ms / sample at 608.5 TPS — the 11.8× / 12× speedup over AR cited in
the paper.

## Install

```bash
# From the Fast-dLLM repo root
pip install -r fast_ddrive/requirements.txt
```

The released checkpoint lives at [`xiwenyoumu/Fast-dDrive`](https://huggingface.co/xiwenyoumu/Fast-dDrive).
The model class and its three decoding paths ship with that repo; every
entry script loads them via `trust_remote_code=True`. There is no local
`fast_ddrive/models/` directory.

## Inference

### Single-shot chat

```bash
# Defaults to --model_path xiwenyoumu/Fast-dDrive (paper checkpoint).
python fast_ddrive/run_chatbot.py \
    --image fast_ddrive/data/example/images/227_CAM_FRONT.jpg \
    --prompt "Describe the driving scene and produce a 5-second plan."
# Override with a local checkpoint:
python fast_ddrive/run_chatbot.py --model_path /path/to/fast_ddrive_ckpt --image example.jpg --prompt "..."
# Add --mode {section_diffusion,scaffold_spec,inference_scaling} to change paths.
```

### Waymo validation eval

```bash
MODEL_PATH=/path/to/fast_ddrive_ckpt \
EVAL_JSON=/path/to/waymo_val.json \
IMAGE_ROOT=/path/to/image_root \
bash fast_ddrive/run_eval.sh
# Defaults: MODE=scaffold_spec (paper canonical SS), NUM_GPUS=auto.
# To try the other paths set MODE=section_diffusion or MODE=inference_scaling.
```

### Inference modes

`run_eval.sh` (and `run_chatbot.py`) accept `--mode`/`MODE` ∈:

| Mode                | Bound method                          | Default threshold | Notes |
|---------------------|----------------------------------------|:-----------------:|-------|
| `section_diffusion` | `mdm_sample_deep_scaffold`             | 0.9               | Pure iterative MDM denoising; no AR verify. |
| `scaffold_spec`     | `scaffold_speculative_sample`          | 0.0               | **Paper canonical SS.** MDM draft + AR verify per block. |
| `inference_scaling` | `scaffold_spec_with_ss_multi_traj`     | 0.0               | SS + shared-prefix multi-trajectory rollouts (defaults to N=4, vt=0.5). |

`scaffold_spec` and `inference_scaling` **must** use
`confidence_threshold=0.0` to reproduce paper numbers; running them at 0.9
silently degrades ADE by ≈1.5% and TPS by ≈30%. The launcher uses the
correct default automatically; override with `--confidence_threshold` only
if you know what you want.

### Official Waymo metrics (ADE / RFS)

`eval/evaluate_waymo_metrics.py` consumes the `predictions.json` written by
`run_eval.sh` and reports ADE@3s, ADE@5s, RFS. It depends on `tensorflow` +
`waymo_open_dataset` — install in a separate env (see `data/README.md`) and
launch via `run_metrics.sh`:

```bash
PRED_JSON=eval_outputs/<ckpt>_scaffold_spec/predictions.json \
GT=/path/to/waymo_val/*.tfrecord*                                  \
PYTHON=/path/to/autovla/bin/python                                 \
bash fast_ddrive/run_metrics.sh
# Output: <dirname PRED_JSON>/waymo_metrics/{waymo_eval_results,waymo_eval_detailed}.json
# GT can also be a pre-computed gt_dict_val.pkl from a previous run.
```

## Training

Finetune Qwen2.5-VL-3B (or any Fast-dDrive checkpoint) with the canonical
SASD recipe (MDM + deep JSON scaffold + section-weighted loss + per-section
Beta noise):

```bash
DATASET_PATH=/path/to/waymo_train.json \
IMAGE_FOLDER=/path/to/image_root \
bash fast_ddrive/train_scripts/train_waymo_sasd.sh
# Defaults: MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct,
#           NUM_TRAIN_EPOCHS=2, LR=1e-5, BATCH=1, GRAD_ACC=4.
```

The launcher mirrors `fast_dvlm/train_scripts/finetune_multimodal_example.sh`:
single DeepSpeed entry, paper-canonical knobs baked in (do not pass
`SECTION_LOSS_WEIGHTS` / `SECTION_NOISE_SCHEDULE` unless you intend to
ablate). Multi-node training is left as user-side SLURM glue; see
`fast_dvlm/` for examples.

## Layout (relative to the Fast-dLLM repo root)

```
Fast-dLLM/
├── third_party/lmflow/                # vendored LMFlow + minimal SASD hooks
└── fast_ddrive/                       # driving-only entry points (this package)
    ├── README.md                      # this file
    ├── requirements.txt, pyproject.toml
    ├── run_chatbot.py                 # single image+prompt demo
    ├── run_eval.sh                    # batch inference launcher
    ├── run_metrics.sh                  # official Waymo ADE / RFS launcher
    ├── eval/
    │   ├── batch_inference.py         # multi-GPU batch inference (3 modes)
    │   ├── evaluate_waymo_metrics.py  # official Waymo ADE / RFS scoring
    │   └── waymo_rfs_utils.py
    ├── train_scripts/
    │   ├── finetune_fast_ddrive.py    # LMFlow finetuner entry (SASD wiring)
    │   └── train_waymo_sasd.sh        # DeepSpeed launcher
    └── data/
        ├── README.md                  # dataset acquisition + JSON schema
        └── example/                   # 2 samples + 6 images for smoke tests
```

The model code itself (`modeling.py`, `configuration.py`, `section_utils.py`,
`generation_utils.py`) ships from the Hugging Face Hub via
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.

## Citation

```bibtex
@misc{zhang2026fastddriveefficientblockdiffusionvlm,
      title={Fast-dDrive: Efficient Block-Diffusion VLM for Autonomous Driving},
      author={Kewei Zhang and Jin Wang and Sensen Gao and Chengyue Wu and Yulong Cao and Songyang Han and Boris Ivanovic and Langechuan Liu and Marco Pavone and Song Han and Daquan Zhou and Enze Xie},
      year={2026},
      eprint={2605.23163},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.23163},
}
```

## Acknowledgements

Built on [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) /
[Fast-dVLM](../fast_dvlm), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL),
[LMFlow](https://github.com/OptimalScale/LMFlow), and the
[Waymo Open Dataset](https://waymo.com/open/).
