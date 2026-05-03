"""Fast-dVLM sglang chatbot with selectable MDM or speculative decoding."""

import argparse
import os
import sys

# Drop this script's dir from sys.path so `import sglang` resolves to the
# installed package, not the local ./sglang source tree (a namespace pkg).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _HERE]

ALGO_MAP = {
    "mdm": "HierarchyBlock",
    "spec": "SpeculativeBlock",
}


def build_inputs(processor, image, prompt):
    from qwen_vl_utils import process_vision_info

    content = []
    if image:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    return inputs.input_ids[0].tolist()


def main():
    parser = argparse.ArgumentParser(description="Fast-dVLM sglang chatbot")
    parser.add_argument(
        "--model-path", default="Efficient-Large-Model/Fast_dVLM_3B",
        help="HuggingFace model id or local path to Fast_dVLM checkpoint.",
    )
    parser.add_argument(
        "--processor-path", default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HF processor for chat template + image preprocessing.",
    )
    parser.add_argument(
        "--algorithm", choices=list(ALGO_MAP.keys()), default="mdm",
        help="mdm = HierarchyBlock (block diffusion); spec = SpeculativeBlock.",
    )
    parser.add_argument("--image", default=None, help="Image URL or local path. Empty for text-only.")
    parser.add_argument("--prompt", default=None, help="Text prompt. If omitted, enters interactive mode.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument(
        "--quantization", default=None,
        choices=["w8a8_fp8"],
        help="Quantization format of the checkpoint. Leave unset for BF16. "
             "w8a8_fp8 requires SM89+ (4090 / L40 / H100 / H200).",
    )
    args = parser.parse_args()

    os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")

    import sglang as sgl
    from transformers import AutoProcessor, AutoTokenizer

    processor = AutoProcessor.from_pretrained(args.processor_path, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer = tokenizer

    dllm_algo = ALGO_MAP[args.algorithm]
    engine_kwargs = dict(
        model_path=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=1,
        chunked_prefill_size=16384,
        dllm_algorithm=dllm_algo,
        disable_cuda_graph=False,
        log_level="warning",
        enable_metrics=True,
        mm_attention_backend="triton_attn",
    )
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization

    print(
        f"Launching sglang Engine with dllm_algorithm={dllm_algo}"
        f"{f', quantization={args.quantization}' if args.quantization else ''} ..."
    )
    engine = sgl.Engine(**engine_kwargs)

    sampling = {"max_new_tokens": args.max_tokens, "temperature": 0.0}

    def run_once(image, prompt):
        input_ids = build_inputs(processor, image, prompt)
        out = engine.generate(
            input_ids=input_ids,
            image_data=[image] if image else None,
            sampling_params=sampling,
        )
        if isinstance(out, list):
            out = out[0]
        return out["text"]

    try:
        if args.prompt:
            image = args.image or "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            print(f"\n{run_once(image, args.prompt)}")
        else:
            print("Interactive mode. Type 'exit' to quit, 'clear' to reset.")
            while True:
                prompt = input("\nYou: ").strip()
                if prompt.lower() == "exit":
                    break
                if prompt.lower() == "clear":
                    print("History cleared.")
                    continue
                if not prompt:
                    continue
                print(f"\nAssistant: {run_once(args.image, prompt)}")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
