"""
Test that SGLang and HuggingFace (no dual cache) produce identical outputs
for Fast dLLM v2.

Tests cover:
  1. With chat template
  2. Without chat template (raw prompt)
"""

import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.test.test_utils import CustomTestCase

DIVIDER_WIDTH = 80
SECTION_CHAR = "="
SUBSECTION_CHAR = "-"

MODEL = "Efficient-Large-Model/Fast_dLLM_v2_7B"
MAX_NEW_TOKENS = 64
BLOCK_SIZE = 32
SMALL_BLOCK_SIZE = 8
THRESHOLD = 0.9

TEST_PROMPTS = [
    "What is 1+1?",
    "What is the capital of France?",
]


def print_section(title: str):
    print("\n" + SECTION_CHAR * DIVIDER_WIDTH)
    print(title)
    print(SECTION_CHAR * DIVIDER_WIDTH)


def print_subsection(title: str):
    print(f"\n{SUBSECTION_CHAR * 40}")
    print(title)
    print(SUBSECTION_CHAR * 40)


def format_with_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_hf_output(model, tokenizer, prompt: str, device):
    """Run HF generation (no dual cache) and return token ids."""
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
            temperature=0.0,
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


def get_sglang_output(engine, tokenizer, prompt: str):
    """Run SGLang generation and return token ids."""
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

    outputs = engine.generate(
        prompt=[prompt],
        sampling_params={
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
        },
    )

    result = outputs[0] if isinstance(outputs, list) else outputs
    token_ids = result["output_ids"]
    full_text = result.get("text", "")

    if full_text.startswith(prompt):
        generated_text = full_text[len(prompt):]
    else:
        generated_text = full_text

    return {
        "prompt_ids": prompt_ids,
        "token_ids": token_ids,
        "text": generated_text,
    }


def compare_outputs(test_case, hf_out, sgl_out, tokenizer, label: str):
    """Compare HF and SGLang outputs, assert exact match."""
    print_subsection(f"Comparison: {label}")

    hf_prompt_ids = hf_out["prompt_ids"]
    sgl_prompt_ids = sgl_out["prompt_ids"]
    print(f"HF  prompt length: {len(hf_prompt_ids)}")
    print(f"SGL prompt length: {len(sgl_prompt_ids)}")

    if hf_prompt_ids != sgl_prompt_ids:
        for i, (h, s) in enumerate(zip(hf_prompt_ids, sgl_prompt_ids)):
            if h != s:
                print(f"  Prompt diff at pos {i}: HF={h}, SGL={s}")
                break
    test_case.assertEqual(
        hf_prompt_ids, sgl_prompt_ids, f"[{label}] Prompt token ids differ"
    )

    hf_tokens = hf_out["token_ids"]
    sgl_tokens = sgl_out["token_ids"]
    print(f"HF  generated: {len(hf_tokens)} tokens  {hf_tokens}")
    print(f"SGL generated: {len(sgl_tokens)} tokens  {sgl_tokens}")
    print(f"HF  text: {repr(hf_out['text'])}")
    print(f"SGL text: {repr(sgl_out['text'])}")

    min_len = min(len(hf_tokens), len(sgl_tokens))
    for i in range(min_len):
        if hf_tokens[i] != sgl_tokens[i]:
            print(
                f"  First token mismatch at pos {i}: "
                f"HF={hf_tokens[i]} ('{tokenizer.decode([hf_tokens[i]])}') vs "
                f"SGL={sgl_tokens[i]} ('{tokenizer.decode([sgl_tokens[i]])}')"
            )
            break

    test_case.assertEqual(
        hf_tokens,
        sgl_tokens,
        f"[{label}] Generated token ids differ!\n"
        f"  HF:  {hf_tokens}\n"
        f"  SGL: {sgl_tokens}",
    )
    print("  EXACT MATCH!")


class TestSGLangHFMatch(CustomTestCase):
    """Test exact match between SGLang and HF (no dual cache) for Fast dLLM v2."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")

        print(f"Loading HF model: {MODEL}")
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        cls.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            dtype=torch.float16,
            trust_remote_code=True,
        ).to(cls.device)
        cls.hf_model.eval()

        print(f"Launching SGLang Engine: {MODEL}")
        cls.sgl_engine = sgl.Engine(
            model_path=MODEL,
            trust_remote_code=True,
            mem_fraction_static=0.85,
            max_running_requests=1,
            attention_backend="flashinfer",
            disable_cuda_graph=True,
            dllm_algorithm="HierarchyBlock",
        )

    @classmethod
    def tearDownClass(cls):
        cls.sgl_engine.shutdown()
        del cls.hf_model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 1. With chat template
    # ------------------------------------------------------------------ #
    def test_with_chat_template(self):
        print_section("With Chat Template (HF no dual cache vs SGLang)")
        for prompt in TEST_PROMPTS:
            formatted = format_with_chat_template(self.tokenizer, prompt)
            print(f"\nRaw prompt: {repr(prompt)}")
            print(f"Formatted:  {repr(formatted)}")

            hf_out = get_hf_output(
                self.hf_model, self.tokenizer, formatted, self.device,
            )
            sgl_out = get_sglang_output(self.sgl_engine, self.tokenizer, formatted)
            compare_outputs(
                self, hf_out, sgl_out, self.tokenizer,
                f"chat_template | {prompt[:40]}",
            )

    # ------------------------------------------------------------------ #
    # 2. Without chat template (raw prompt)
    # ------------------------------------------------------------------ #
    def test_without_chat_template(self):
        print_section("Without Chat Template (HF no dual cache vs SGLang)")
        for prompt in TEST_PROMPTS:
            print(f"\nRaw prompt: {repr(prompt)}")

            hf_out = get_hf_output(
                self.hf_model, self.tokenizer, prompt, self.device,
            )
            sgl_out = get_sglang_output(self.sgl_engine, self.tokenizer, prompt)
            compare_outputs(
                self, hf_out, sgl_out, self.tokenizer,
                f"raw_prompt | {prompt[:40]}",
            )


if __name__ == "__main__":
    unittest.main()
