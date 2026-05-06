"""Fast-dVLM command-line chatbot with speculative block-causal decoding."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def build_messages(image, prompt):
    content = []
    if image:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def run_inference(model, processor, image, prompt, args):
    messages = build_messages(image, prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    mask_id = processor.tokenizer.encode(args.mask_token)[0]

    gen_kwargs = {
        "input_ids": inputs.input_ids,
        "tokenizer": processor.tokenizer,
        "block_size": args.block_size,
        "max_tokens": args.max_tokens,
        "mask_id": mask_id,
    }
    if hasattr(inputs, "pixel_values"):
        gen_kwargs["pixel_values"] = inputs.pixel_values
    if hasattr(inputs, "image_grid_thw"):
        gen_kwargs["image_grid_thw"] = inputs.image_grid_thw

    generated_ids = model.generate(**gen_kwargs)
    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
    )[0]


def main():
    parser = argparse.ArgumentParser(description="Fast-dVLM chatbot")
    parser.add_argument(
        "--model-name", default="output_models/Fast_dVLM_3B",
        help="HuggingFace model id or local path.",
    )
    parser.add_argument(
        "--image", default=None,
        help="Image URL or local path. Leave empty for text-only.",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Text prompt. If omitted, enters interactive mode.",
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--mask-token", default="|<MASK>|")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=False)
    processor.tokenizer = tokenizer

    if args.prompt:
        image = args.image or "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        output = run_inference(model, processor, image, args.prompt, args)
        print(f"\n{output}")
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
            image = args.image
            output = run_inference(model, processor, image, prompt, args)
            print(f"\nAssistant: {output}")


if __name__ == "__main__":
    main()
