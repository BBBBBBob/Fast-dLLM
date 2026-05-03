"""
Compare HF vs SGLang token-by-token on GSM8K 5-shot + simple questions.
Print all tokens with ANSI color highlighting at first divergence.
Output 2 txt files: with_template.txt and without_template.txt

Usage:
    CUDA_VISIBLE_DEVICES=4 python test_token_diff_colored.py
"""

import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.utils import download_and_cache_file, read_jsonl

# ── Constants (HF & SGLang IDENTICAL) ──────────────────────────────────────
MODEL = "Efficient-Large-Model/Fast_dLLM_v2_7B"
MAX_NEW_TOKENS = 512
BLOCK_SIZE = 32
SMALL_BLOCK_SIZE = 8
THRESHOLD = 0.9
TEMPERATURE = 0.0
NUM_SHOTS = 5
NUM_QUESTIONS = 5

SIMPLE_QUESTIONS = [
    "What is 1+1?",
    "What is the capital of France?",
    "What color is the sky?",
    "How many days are in a week?",
    "What is 2 * 3 + 4?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet in our solar system?",
    "If I have 10 apples and give away 3, how many do I have?",
]

# ── ANSI colors ────────────────────────────────────────────────────────────
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
RESET = "\033[0m"
DIM = "\033[2m"


# ── GSM8K helpers ──────────────────────────────────────────────────────────
def load_gsm8k():
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    return list(read_jsonl(download_and_cache_file(url)))


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
    few_shot = get_few_shot_examples(lines, num_shots)
    return [few_shot + get_one_example(lines, i, False) for i in range(num_questions)]


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
    return {"prompt_ids": inputs["input_ids"][0].tolist(), "token_ids": new_ids, "text": text}


# ── SGLang generation ─────────────────────────────────────────────────────
def sglang_generate(engine, tokenizer, prompt):
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    outputs = engine.generate(
        prompt=[prompt],
        sampling_params={"max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE},
    )
    result = outputs[0] if isinstance(outputs, list) else outputs
    token_ids = result["output_ids"]
    full_text = result.get("text", "")
    gen_text = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
    return {"prompt_ids": prompt_ids, "token_ids": token_ids, "text": gen_text}


# ── Pretty print with colors ──────────────────────────────────────────────
def print_colored_comparison(hf_out, sgl_out, tokenizer, label):
    """Print token-by-token comparison with ANSI colors. Returns result dict."""
    hf_ids = hf_out["token_ids"]
    sgl_ids = sgl_out["token_ids"]
    max_len = max(len(hf_ids), len(sgl_ids))
    min_len = min(len(hf_ids), len(sgl_ids))

    # Find first diff
    first_diff = None
    for i in range(min_len):
        if hf_ids[i] != sgl_ids[i]:
            first_diff = i
            break
    if first_diff is None and len(hf_ids) != len(sgl_ids):
        first_diff = min_len

    print(f"\n{BOLD}{CYAN}{'=' * 100}{RESET}")
    print(f"{BOLD}{CYAN}  [{label}]{RESET}")
    print(f"{CYAN}{'=' * 100}{RESET}")

    # Prompt check
    if hf_out["prompt_ids"] == sgl_out["prompt_ids"]:
        print(f"  Prompt: {GREEN}IDENTICAL{RESET} ({len(hf_out['prompt_ids'])} tokens)")
    else:
        print(f"  Prompt: {RED}DIFFER{RESET}")

    if first_diff is None:
        print(f"  Result: {BOLD}{GREEN}ALL {len(hf_ids)} TOKENS IDENTICAL{RESET}")
    else:
        matches = sum(1 for i in range(min_len) if hf_ids[i] == sgl_ids[i])
        print(f"  Result: {BOLD}{RED}FIRST DIVERGENCE AT POSITION {first_diff}{RESET}  "
              f"(match: {matches}/{min_len} = {matches/max(min_len,1)*100:.1f}%)")

    print(f"  HF  generated: {len(hf_ids)} tokens")
    print(f"  SGL generated: {len(sgl_ids)} tokens")

    # ── Print HF tokens ──
    print(f"\n  {BOLD}HF  tokens:{RESET}")
    _print_token_line(hf_ids, sgl_ids, tokenizer, first_diff, "hf")

    # ── Print SGLang tokens ──
    print(f"\n  {BOLD}SGL tokens:{RESET}")
    _print_token_line(sgl_ids, hf_ids, tokenizer, first_diff, "sgl")

    # ── Print text ──
    print(f"\n  {BOLD}HF  text:{RESET}")
    _print_text_colored(hf_out["text"], sgl_out["text"])
    print(f"\n  {BOLD}SGL text:{RESET}")
    _print_text_colored(sgl_out["text"], hf_out["text"])

    return {
        "label": label,
        "hf_len": len(hf_ids),
        "sgl_len": len(sgl_ids),
        "first_diff_pos": first_diff,
        "hf_ids": hf_ids,
        "sgl_ids": sgl_ids,
        "hf_text": hf_out["text"],
        "sgl_text": sgl_out["text"],
    }


def _print_token_line(my_ids, other_ids, tokenizer, first_diff, side):
    """Print tokens with color: green=match, red bg=first diff, yellow=subsequent diff."""
    parts = []
    for i, tid in enumerate(my_ids):
        tok_str = tokenizer.decode([tid]).replace("\n", "\\n")
        display = f"[{i}]{tid}:'{tok_str}'"

        if i < len(other_ids):
            if my_ids[i] == other_ids[i]:
                # Match
                if first_diff is not None and i < first_diff:
                    parts.append(f"{GREEN}{display}{RESET}")
                else:
                    parts.append(f"{DIM}{display}{RESET}")
            elif i == first_diff:
                # First divergence - bright red background
                parts.append(f"{BOLD}{BG_RED} >>> {display} <<< {RESET}")
            else:
                # Subsequent diff
                parts.append(f"{YELLOW}{display}{RESET}")
        else:
            # Extra tokens beyond other's length
            parts.append(f"{DIM}{display}{RESET}")

    # Print in rows of ~5 tokens for readability
    ROW_SIZE = 5
    for row_start in range(0, len(parts), ROW_SIZE):
        row = parts[row_start:row_start + ROW_SIZE]
        print(f"    {' '.join(row)}")


def _print_text_colored(my_text, other_text):
    """Print text, coloring the common prefix green and the rest red."""
    common_len = 0
    for i in range(min(len(my_text), len(other_text))):
        if my_text[i] == other_text[i]:
            common_len = i + 1
        else:
            break

    if common_len > 0:
        print(f"    {GREEN}{repr(my_text[:common_len])}{RESET}", end="")
    if common_len < len(my_text):
        print(f"{RED}{repr(my_text[common_len:])}{RESET}", end="")
    print()


# ── TXT output (no ANSI) ──────────────────────────────────────────────────
def write_txt_report(results, filename, title):
    """Write a plain-text report with all tokens and marked divergence."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 100}\n")
        f.write(f"  {title}\n")
        f.write(f"  Model: {MODEL}\n")
        f.write(f"  Settings: max_new_tokens={MAX_NEW_TOKENS}, block_size={BLOCK_SIZE}, "
                f"small_block_size={SMALL_BLOCK_SIZE}, threshold={THRESHOLD}, "
                f"temperature={TEMPERATURE}\n")
        f.write(f"{'=' * 100}\n\n")

        for r in results:
            diff = r["first_diff_pos"]
            diff_str = str(diff) if diff is not None else "NONE (all identical)"
            hf_ids = r["hf_ids"]
            sgl_ids = r["sgl_ids"]
            min_len = min(len(hf_ids), len(sgl_ids))
            matches = sum(1 for i in range(min_len) if hf_ids[i] == sgl_ids[i])

            f.write(f"{'=' * 100}\n")
            f.write(f"[{r['label']}]\n")
            f.write(f"  HF  generated: {r['hf_len']} tokens\n")
            f.write(f"  SGL generated: {r['sgl_len']} tokens\n")
            f.write(f"  First divergence position: {diff_str}\n")
            if diff is not None:
                f.write(f"  Match rate: {matches}/{min_len} = {matches/max(min_len,1)*100:.1f}%\n")
            f.write(f"\n")

            # Token-by-token table
            f.write(f"  {'Pos':<6} {'Match':<8} {'HF ID':<10} {'HF Token':<25} {'SGL ID':<10} {'SGL Token':<25}\n")
            f.write(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*25} {'-'*10} {'-'*25}\n")

            max_len = max(len(hf_ids), len(sgl_ids))
            for i in range(max_len):
                hf_id_str = str(hf_ids[i]) if i < len(hf_ids) else "---"
                sgl_id_str = str(sgl_ids[i]) if i < len(sgl_ids) else "---"
                hf_tok_str = repr(tokenizer_global.decode([hf_ids[i]])) if i < len(hf_ids) else "---"
                sgl_tok_str = repr(tokenizer_global.decode([sgl_ids[i]])) if i < len(sgl_ids) else "---"

                if i < len(hf_ids) and i < len(sgl_ids) and hf_ids[i] == sgl_ids[i]:
                    match_marker = "  OK  "
                elif i == diff:
                    match_marker = ">>DIFF"
                else:
                    match_marker = "  diff"

                f.write(f"  {i:<6} {match_marker:<8} {hf_id_str:<10} {hf_tok_str:<25} {sgl_id_str:<10} {sgl_tok_str:<25}\n")

            f.write(f"\n  HF  full text:\n")
            for line in r["hf_text"].split("\n"):
                f.write(f"    {line}\n")
            f.write(f"\n  SGL full text:\n")
            for line in r["sgl_text"].split("\n"):
                f.write(f"    {line}\n")
            f.write(f"\n\n")

        # Summary table
        f.write(f"\n{'=' * 100}\n")
        f.write(f"SUMMARY\n")
        f.write(f"{'=' * 100}\n")
        f.write(f"{'Label':<50} {'HF len':>8} {'SGL len':>8} {'1st diff':>10} {'Match%':>8}\n")
        f.write(f"{'-' * 90}\n")
        for r in results:
            d = str(r["first_diff_pos"]) if r["first_diff_pos"] is not None else "SAME"
            min_l = min(r["hf_len"], r["sgl_len"])
            matches = sum(1 for i in range(min_l) if r["hf_ids"][i] == r["sgl_ids"][i])
            pct = matches / max(min_l, 1) * 100
            f.write(f"{r['label']:<50} {r['hf_len']:>8} {r['sgl_len']:>8} {d:>10} {pct:>7.1f}%\n")


# ── Main ───────────────────────────────────────────────────────────────────
tokenizer_global = None


def main():
    global tokenizer_global
    device = torch.device("cuda")
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Load data
    print(f"{BOLD}Loading GSM8K...{RESET}")
    lines = load_gsm8k()
    raw_gsm8k = build_few_shot_prompts(lines, NUM_SHOTS, NUM_QUESTIONS)
    gsm8k_labels = [lines[i]["question"][:60] for i in range(NUM_QUESTIONS)]

    # Load models
    print(f"{BOLD}Loading HF model: {MODEL}{RESET}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer_global = tokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    hf_model.eval()

    print(f"{BOLD}Launching SGLang Engine: {MODEL}{RESET}")
    sgl_engine = sgl.Engine(
        model_path=MODEL,
        trust_remote_code=True,
        mem_fraction_static=0.85,
        max_running_requests=1,
        attention_backend="flashinfer",
        disable_cuda_graph=True,
        dllm_algorithm="HierarchyBlock",
    )

    # ── WITH template ──────────────────────────────────────────────────
    results_with_tmpl = []

    print(f"\n{BOLD}{BG_GREEN} WITH CHAT TEMPLATE - GSM8K 5-shot {RESET}\n")
    for idx, raw_prompt in enumerate(raw_gsm8k):
        formatted = apply_chat_template(tokenizer, raw_prompt)
        hf_out = hf_generate(hf_model, tokenizer, formatted, device)
        sgl_out = sglang_generate(sgl_engine, tokenizer, formatted)
        info = print_colored_comparison(hf_out, sgl_out, tokenizer,
                                        f"GSM8K Q{idx} (template): {gsm8k_labels[idx]}")
        results_with_tmpl.append(info)

    print(f"\n{BOLD}{BG_GREEN} WITH CHAT TEMPLATE - Simple Questions {RESET}\n")
    for idx, q in enumerate(SIMPLE_QUESTIONS):
        formatted = apply_chat_template(tokenizer, q)
        hf_out = hf_generate(hf_model, tokenizer, formatted, device)
        sgl_out = sglang_generate(sgl_engine, tokenizer, formatted)
        info = print_colored_comparison(hf_out, sgl_out, tokenizer,
                                        f"Simple Q{idx} (template): {q}")
        results_with_tmpl.append(info)

    # ── WITHOUT template ───────────────────────────────────────────────
    results_no_tmpl = []

    print(f"\n{BOLD}{BG_YELLOW} WITHOUT CHAT TEMPLATE - GSM8K 5-shot {RESET}\n")
    for idx, raw_prompt in enumerate(raw_gsm8k):
        hf_out = hf_generate(hf_model, tokenizer, raw_prompt, device)
        sgl_out = sglang_generate(sgl_engine, tokenizer, raw_prompt)
        info = print_colored_comparison(hf_out, sgl_out, tokenizer,
                                        f"GSM8K Q{idx} (no tmpl): {gsm8k_labels[idx]}")
        results_no_tmpl.append(info)

    print(f"\n{BOLD}{BG_YELLOW} WITHOUT CHAT TEMPLATE - Simple Questions {RESET}\n")
    for idx, q in enumerate(SIMPLE_QUESTIONS):
        hf_out = hf_generate(hf_model, tokenizer, q, device)
        sgl_out = sglang_generate(sgl_engine, tokenizer, q)
        info = print_colored_comparison(hf_out, sgl_out, tokenizer,
                                        f"Simple Q{idx} (no tmpl): {q}")
        results_no_tmpl.append(info)

    # ── Write TXT files ────────────────────────────────────────────────
    txt_with = os.path.join(out_dir, "with_template.txt")
    txt_no = os.path.join(out_dir, "without_template.txt")
    write_txt_report(results_with_tmpl, txt_with,
                     "HF vs SGLang Token Comparison — WITH Chat Template")
    write_txt_report(results_no_tmpl, txt_no,
                     "HF vs SGLang Token Comparison — WITHOUT Chat Template")
    print(f"\n{BOLD}{GREEN}Saved: {txt_with}{RESET}")
    print(f"{BOLD}{GREEN}Saved: {txt_no}{RESET}")

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{BOLD}{'=' * 100}{RESET}")
    print(f"{BOLD}  FINAL SUMMARY{RESET}")
    print(f"{BOLD}{'=' * 100}{RESET}")
    for tag, results in [("WITH template", results_with_tmpl), ("WITHOUT template", results_no_tmpl)]:
        print(f"\n  {BOLD}{tag}:{RESET}")
        print(f"  {'Label':<55} {'HF':>6} {'SGL':>6} {'1st diff':>10} {'Match%':>8}")
        print(f"  {'-' * 90}")
        for r in results:
            d = str(r["first_diff_pos"]) if r["first_diff_pos"] is not None else "SAME"
            min_l = min(r["hf_len"], r["sgl_len"])
            matches = sum(1 for i in range(min_l) if r["hf_ids"][i] == r["sgl_ids"][i])
            pct = matches / max(min_l, 1) * 100
            color = GREEN if d == "SAME" else (YELLOW if pct > 80 else RED)
            print(f"  {r['label']:<55} {r['hf_len']:>6} {r['sgl_len']:>6} {color}{d:>10}{RESET} {color}{pct:>7.1f}%{RESET}")

    # Cleanup
    sgl_engine.shutdown()
    del hf_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
