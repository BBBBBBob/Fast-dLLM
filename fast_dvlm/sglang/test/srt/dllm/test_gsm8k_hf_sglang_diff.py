"""
Compare HF vs SGLang on 5 GSM8K samples (5-shot), with and without chat template.
Find the first token position where outputs diverge.

Usage:
    CUDA_VISIBLE_DEVICES=4 python test_gsm8k_hf_sglang_diff.py
"""

import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.utils import download_and_cache_file, read_jsonl

# ── Shared constants (keep HF & SGLang IDENTICAL) ──────────────────────────
MODEL = "Efficient-Large-Model/Fast_dLLM_v2_7B"
MAX_NEW_TOKENS = 512
BLOCK_SIZE = 32
SMALL_BLOCK_SIZE = 8
THRESHOLD = 0.9
TEMPERATURE = 0.0
NUM_SHOTS = 5
NUM_QUESTIONS = 5

DIVIDER = "=" * 90
SUB_DIVIDER = "-" * 60


# ── GSM8K data helpers ─────────────────────────────────────────────────────
def load_gsm8k():
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    filename = download_and_cache_file(url)
    return list(read_jsonl(filename))


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def build_few_shot_prompts(lines, num_shots, num_questions):
    """Return list of (raw_prompt, label) for the first num_questions."""
    few_shot = get_few_shot_examples(lines, num_shots)
    prompts = []
    for i in range(num_questions):
        q = get_one_example(lines, i, False)
        prompts.append(few_shot + q)
    return prompts


# ── Chat template ──────────────────────────────────────────────────────────
def apply_chat_template(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


# ── HF generation ─────────────────────────────────────────────────────────
def hf_generate(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            block_size=BLOCK_SIZE,
            small_block_size=SMALL_BLOCK_SIZE,
            threshold=THRESHOLD,
            temperature=TEMPERATURE,
            use_block_cache=False,
        )

    full_ids = outputs[0].tolist()
    new_ids = full_ids[prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=False)
    return {
        "prompt_ids": inputs["input_ids"][0].tolist(),
        "token_ids": new_ids,
        "text": text,
    }


# ── SGLang generation ─────────────────────────────────────────────────────
def sglang_generate(engine, tokenizer, prompt):
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

    outputs = engine.generate(
        prompt=[prompt],
        sampling_params={
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
        },
    )

    result = outputs[0] if isinstance(outputs, list) else outputs
    token_ids = result["output_ids"]
    full_text = result.get("text", "")
    if full_text.startswith(prompt):
        gen_text = full_text[len(prompt):]
    else:
        gen_text = full_text

    return {
        "prompt_ids": prompt_ids,
        "token_ids": token_ids,
        "text": gen_text,
    }


# ── Comparison ─────────────────────────────────────────────────────────────
def compare_tokens(hf_out, sgl_out, tokenizer, label):
    """Compare and print first divergence position. Return diff info dict."""
    hf_ids = hf_out["token_ids"]
    sgl_ids = sgl_out["token_ids"]
    min_len = min(len(hf_ids), len(sgl_ids))

    first_diff = None
    for i in range(min_len):
        if hf_ids[i] != sgl_ids[i]:
            first_diff = i
            break

    if first_diff is None and len(hf_ids) != len(sgl_ids):
        first_diff = min_len  # lengths differ

    matches = sum(1 for i in range(min_len) if hf_ids[i] == sgl_ids[i])

    print(f"\n{SUB_DIVIDER}")
    print(f"  [{label}]")
    print(f"  HF  prompt len: {len(hf_out['prompt_ids'])}, generated: {len(hf_ids)} tokens")
    print(f"  SGL prompt len: {len(sgl_out['prompt_ids'])}, generated: {len(sgl_ids)} tokens")

    # Check prompt match
    if hf_out["prompt_ids"] != sgl_out["prompt_ids"]:
        for j, (h, s) in enumerate(zip(hf_out["prompt_ids"], sgl_out["prompt_ids"])):
            if h != s:
                print(f"  !! Prompt IDs differ at pos {j}: HF={h} vs SGL={s}")
                break
    else:
        print(f"  Prompt IDs: IDENTICAL ({len(hf_out['prompt_ids'])} tokens)")

    if first_diff is None:
        print(f"  Output tokens: ALL IDENTICAL ({len(hf_ids)} tokens)")
    else:
        print(f"  Token match rate: {matches}/{min_len} ({matches/max(min_len,1)*100:.1f}%)")
        print(f"  ** First divergence at position {first_diff} **")
        if first_diff < min_len:
            hf_tok = hf_ids[first_diff]
            sgl_tok = sgl_ids[first_diff]
            print(f"     HF  token[{first_diff}] = {hf_tok} ('{tokenizer.decode([hf_tok])}')")
            print(f"     SGL token[{first_diff}] = {sgl_tok} ('{tokenizer.decode([sgl_tok])}')")

        # Show context around divergence
        start = max(0, first_diff - 2)
        end = min(min_len, first_diff + 5)
        print(f"  Context tokens [{start}:{end}]:")
        print(f"     HF:  {hf_ids[start:end]}")
        print(f"     SGL: {sgl_ids[start:end]}")

    print(f"  HF  text (first 200 chars): {repr(hf_out['text'][:200])}")
    print(f"  SGL text (first 200 chars): {repr(sgl_out['text'][:200])}")

    return {
        "label": label,
        "hf_len": len(hf_ids),
        "sgl_len": len(sgl_ids),
        "first_diff_pos": first_diff,
        "match_rate": matches / max(min_len, 1),
    }


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda")

    # Load GSM8K
    print(f"{DIVIDER}\nLoading GSM8K data...\n{DIVIDER}")
    lines = load_gsm8k()
    raw_prompts = build_few_shot_prompts(lines, NUM_SHOTS, NUM_QUESTIONS)
    print(f"Built {len(raw_prompts)} prompts with {NUM_SHOTS}-shot examples")

    # Load HF model
    print(f"\n{DIVIDER}\nLoading HF model: {MODEL}\n{DIVIDER}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    hf_model.eval()

    # Launch SGLang engine
    print(f"\n{DIVIDER}\nLaunching SGLang Engine: {MODEL}\n{DIVIDER}")
    sgl_engine = sgl.Engine(
        model_path=MODEL,
        trust_remote_code=True,
        mem_fraction_static=0.85,
        max_running_requests=1,
        attention_backend="flashinfer",
        disable_cuda_graph=True,
        dllm_algorithm="HierarchyBlock",
    )

    all_results = []

    # ────────────────────────────────────────────────────────────────────
    # Test 1: WITHOUT chat template (raw few-shot prompt)
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'#' * 90}")
    print(f"# TEST 1: GSM8K 5-shot — WITHOUT chat template")
    print(f"{'#' * 90}")

    for idx, raw_prompt in enumerate(raw_prompts):
        print(f"\n{DIVIDER}\nQuestion {idx} (no template)\n{DIVIDER}")
        print(f"Question: {lines[idx]['question'][:100]}...")

        hf_out = hf_generate(hf_model, tokenizer, raw_prompt, device)
        sgl_out = sglang_generate(sgl_engine, tokenizer, raw_prompt)
        info = compare_tokens(hf_out, sgl_out, tokenizer, f"Q{idx}_no_template")
        all_results.append(info)

    # ────────────────────────────────────────────────────────────────────
    # Test 2: WITH chat template
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'#' * 90}")
    print(f"# TEST 2: GSM8K 5-shot — WITH chat template")
    print(f"{'#' * 90}")

    for idx, raw_prompt in enumerate(raw_prompts):
        formatted = apply_chat_template(tokenizer, raw_prompt)
        print(f"\n{DIVIDER}\nQuestion {idx} (with template)\n{DIVIDER}")
        print(f"Question: {lines[idx]['question'][:100]}...")

        hf_out = hf_generate(hf_model, tokenizer, formatted, device)
        sgl_out = sglang_generate(sgl_engine, tokenizer, formatted)
        info = compare_tokens(hf_out, sgl_out, tokenizer, f"Q{idx}_with_template")
        all_results.append(info)

    # ────────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'#' * 90}")
    print(f"# SUMMARY")
    print(f"{'#' * 90}")
    print(f"\nSettings: model={MODEL}, max_new_tokens={MAX_NEW_TOKENS}, "
          f"block_size={BLOCK_SIZE}, small_block_size={SMALL_BLOCK_SIZE}, "
          f"threshold={THRESHOLD}, temperature={TEMPERATURE}")
    print(f"\n{'Label':<25} {'HF len':>8} {'SGL len':>8} {'1st diff':>10} {'Match%':>8}")
    print("-" * 65)
    for r in all_results:
        diff_str = str(r["first_diff_pos"]) if r["first_diff_pos"] is not None else "SAME"
        print(f"{r['label']:<25} {r['hf_len']:>8} {r['sgl_len']:>8} {diff_str:>10} {r['match_rate']*100:>7.1f}%")

    # Cleanup
    sgl_engine.shutdown()
    del hf_model
    torch.cuda.empty_cache()

    # Save results to JSON
    output_file = os.path.join(os.path.dirname(__file__), "gsm8k_diff_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
