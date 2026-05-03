# Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM

[![Project](https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages)](https://nvlabs.github.io/Fast-dLLM/fast_dvlm/)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2604.06832)
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/Efficient-Large-Model/Fast_dVLM_3B)

Fast-dVLM is a block-diffusion-based Vision-Language Model (VLM) that enables **KV-cache-compatible parallel decoding** and **speculative block decoding** for inference acceleration. Built on **Qwen2.5-VL-3B-Instruct**, Fast-dVLM directly converts the pretrained AR VLM into a block-diffusion model in a single stage.

## Key Highlights

- **Lossless Quality**: Matches the AR baseline (Qwen2.5-VL-3B) across **11 multimodal benchmarks** (74.0 avg).
- **Up to 6.18x Speedup**: With SGLang integration and FP8 quantization.
- **2.63x Tokens/NFE**: With self-speculative block decoding.
- **Direct Conversion**: Single-stage AR-to-diffusion conversion outperforms two-stage approach (73.3 vs 60.2 avg).

## Key Techniques

- **Block-Size Annealing**: Curriculum that progressively increases the block size during training.
- **Causal Context Attention**: Noisy tokens attend bidirectionally within blocks (N2N), to clean tokens from preceding blocks (N2C), while clean tokens follow causal attention (C2C).
- **Auto-Truncation Masking**: Prevents cross-turn leakage in multi-turn dialogue.
- **Vision-Efficient Concatenation**: Vision embeddings included only in the clean stream, reducing peak memory by 15% and training time by 14.2%.

## Benchmark Results

| Model | AI2D | ChartQA | DocVQA | GQA | MMBench | MMMU | POPE | RWQA | SEED2+ | TextVQA | Avg | Tok/NFE |
|-------|------|---------|--------|-----|---------|------|------|------|--------|---------|-----|---------|
| Qwen2.5-VL-3B | 80.8 | 84.0 | 93.1 | 59.0 | 76.9 | 47.3 | 86.2 | 65.1 | 68.6 | 79.1 | 74.0 | 1.00 |
| **Fast-dVLM (MDM)** | 79.7 | 82.8 | 92.1 | 63.0 | 74.2 | 44.6 | 88.6 | 65.1 | 67.2 | 76.1 | 73.3 | 1.95 |
| **Fast-dVLM (spec.)** | 79.7 | 83.1 | 92.9 | 63.3 | 74.3 | 46.6 | 88.6 | 65.1 | 67.2 | 79.3 | **74.0** | **2.63** |

### Inference Acceleration

| Setting | MMMU-Pro-V | TPS | SpeedUp |
|---------|------------|-----|---------|
| AR baseline | 26.3 | 56.7 | 1.00x |
| Fast-dVLM (MDM, τ=0.9) | 21.4 | 82.2 | 1.45x |
| + Spec. decoding (linear) | 24.6 | 112.7 | 1.98x |
| + SGLang serving | 24.1 | 319.0 | 5.63x |
| + SmoothQuant-W8A8 (FP8) | 23.8 | **350.3** | **6.18x** |

## Quick Start

### Installation

```bash
cd fast_dvlm
pip install -r requirements.txt
```

### Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model_name = "Efficient-Large-Model/Fast_dVLM_3B"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
processor.tokenizer = tokenizer

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text], images=image_inputs, videos=video_inputs,
    padding=True, return_tensors="pt",
).to(model.device)

mask_id = tokenizer.encode("|<MASK>|")[0]

generated_ids = model.generate(
    input_ids=inputs.input_ids,
    tokenizer=tokenizer,
    pixel_values=inputs.pixel_values,
    image_grid_thw=inputs.image_grid_thw,
    mask_id=mask_id,
    max_tokens=512,
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### Command Line Chatbot

Checkpoint is `--model-name` (default `Efficient-Large-Model/Fast_dVLM_3B`): HuggingFace repo id or a local path, same as `MODEL_PATH` for `run_eval.sh`.

```bash
# Single query
python run_chatbot.py --prompt "Describe this image." --image path/to/image.jpg

# Interactive mode
python run_chatbot.py --image path/to/image.jpg
```

Commands in interactive mode:
- Type your message and press Enter
- `clear` - Clear conversation history
- `exit` - Quit the chatbot

## Evaluation (VLMEvalKit)

[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) is **vendored** at `../third_party/VLMEvalKit` (i.e. `Fast-dLLM/third_party/VLMEvalKit`). `run_eval.sh` runs one dataset per invocation; **default `TASK` is `DocVQA_VAL`** as a concrete example—override with `TASK=…` for any other VLMEval split.

From the **Fast-dLLM repository root**:

```bash
pip install -r fast_dvlm/requirements.txt
pip install -e third_party/VLMEvalKit
```

Example (DocVQA val split by default). Use the **same checkpoint** as the chatbot: HuggingFace id or local directory for `run_chatbot.py --model-name` (default `Efficient-Large-Model/Fast_dVLM_3B`).

```bash
bash fast_dvlm/run_eval.sh --help
MODEL_PATH=Efficient-Large-Model/Fast_dVLM_3B bash fast_dvlm/run_eval.sh
# Local tree: MODEL_PATH=/path/to/Fast_dVLM_3B bash fast_dvlm/run_eval.sh
# Other split: TASK=MMBench_DEV_EN_V11 DATASET_CLASS=ImageMCQDataset MODEL_PATH=… bash fast_dvlm/run_eval.sh
```

Inference uses the checkpoint’s own `generate` in `modeling.py` (`trust_remote_code` + `AutoModelForCausalLM`), same stack as `run_chatbot.py`. This folder only adds `vlmeval_run.py` (VLMEval config + thin wrapper) and `run_eval.sh`. If weights live in a folder without a processor, set `PROCESSOR_PATH` (e.g. `Qwen/Qwen2.5-VL-3B-Instruct`); otherwise the processor is loaded from `MODEL_PATH` like the chatbot.

To refresh VLMEvalKit, replace `third_party/VLMEvalKit` and commit.

## SGLang-Accelerated Inference

Fast-dVLM ships with a customized SGLang fork that implements two dLLM algorithms:

- **`HierarchyBlock`** — block-diffusion parallel decoding (MDM mode)
- **`SpeculativeBlock`** — self-speculative block decoding (≈2.6× tokens/NFE)

### Install

```bash
cd fast_dvlm/sglang/python
pip install -e .
```

The install pulls in SGLang's native dependencies (flashinfer, sgl-kernel, transformers, etc.). Use a dedicated conda env to avoid version conflicts.

### Command Line Chatbot (SGLang)

```bash
# MDM (HierarchyBlock)
python run_chatbot_sglang.py --algorithm mdm --prompt "Describe this image." --image path/to/image.jpg

# Speculative block decoding
python run_chatbot_sglang.py --algorithm spec --prompt "Describe this image." --image path/to/image.jpg

# Interactive mode
python run_chatbot_sglang.py --algorithm spec --image path/to/image.jpg
```

Key flags:
- `--algorithm {mdm,spec}` — select MDM (HierarchyBlock) or speculative decoding (SpeculativeBlock)
- `--model-path` — HF id or local path (default `Efficient-Large-Model/Fast_dVLM_3B`)
- `--processor-path` — HF processor for chat template + image preprocessing (default `Qwen/Qwen2.5-VL-3B-Instruct`)
- `--max-tokens`, `--mem-fraction-static` — generation length / GPU memory budget
- `--quantization w8a8_fp8` — load the FP8 checkpoint (see below)

If you hit a CuDNN/PyTorch 2.9 compatibility warning, set `SGLANG_DISABLE_CUDNN_CHECK=1` in the environment before launch.

### FP8 Quantized Checkpoint

We provide a SmoothQuant-W8A8 FP8 checkpoint for the 6.18× speedup reported above:

- [`Sensen02/Fast_dVLM_3B_W8A8_FP8`](https://huggingface.co/Sensen02/Fast_dVLM_3B_W8A8_FP8) — language tower in FP8 (E4M3), visual encoder kept in BF16.

Hardware requirement: **SM89+** (RTX 4090 / L40 / H100 / H200). Earlier GPUs (A100, V100) do not have FP8 tensor cores and are not supported.

Launch with `--quantization`:

```bash
# FP8 inference (requires SM89+)
python run_chatbot_sglang.py \
    --algorithm spec \
    --model-path Sensen02/Fast_dVLM_3B_W8A8_FP8 \
    --quantization w8a8_fp8 \
    --prompt "Describe this image." \
    --image path/to/image.jpg
```

The quantized checkpoint ships with a `quantization_config` entry in `config.json`:

```json
"quantization_config": {
    "quant_method": "w8a8_fp8",
    "is_dynamic": false,
    "ignore": ["re:visual.*"]
}
```

SGLang reads this automatically:
- Layers matching `ignore` (visual encoder) stay in BF16.
- Remaining linear layers use per-channel static FP8 weights + per-token dynamic FP8 activations.

> Running on H100 (SM90)? Diffusion decoding produces short token blocks; the CUTLASS TMA kernel requires ≥64 rows, so we fall back to the Triton FP8 GEMM for short batches automatically. No extra flags needed.

## File Structure

```
Fast-dLLM/
├── third_party/
│   └── VLMEvalKit/
└── fast_dvlm/
    ├── README.md
    ├── requirements.txt
    ├── run_chatbot.py
    ├── run_chatbot_sglang.py   # SGLang-backed chatbot (MDM + speculative)
    ├── sglang/                 # Customized SGLang with Fast-dVLM model + dLLM algorithms
    ├── vlmeval_run.py          # config + VLMEval hook (calls ckpt ``generate``)
    └── run_eval.sh             # VLMEval driver (default TASK=DocVQA_VAL)
```

## Citation

```bibtex
@misc{wu2026fastdvlmefficientblockdiffusionvlm,
      title={Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM},
      author={Chengyue Wu and Shiyi Lan and Yonggan Fu and Sensen Gao and Jin Wang and Jincheng Yu and Jose M. Alvarez and Pavlo Molchanov and Ping Luo and Song Han and Ligeng Zhu and Enze Xie},
      year={2026},
      eprint={2604.06832},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.06832},
}
```

## Acknowledgements

We thank [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) for the base model architecture.
