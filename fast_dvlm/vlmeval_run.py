#!/usr/bin/env python3
"""VLMEvalKit: write JSON config + register ``Fast_dVLM`` + delegate to VLMEvalKit ``run.py``.

  python vlmeval_run.py write-config   # env: CFG_PATH, MODEL_PATH_ABS, DATASETS, …
  torchrun … vlmeval_run.py --config … --work-dir …
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Union


def write_vlmeval_config() -> None:
    """Env: CFG_PATH, MODEL_PATH_ABS; optional DATASETS, DATASET_CLASS, PROCESSOR_PATH (defaults to MODEL_PATH_ABS), MAX_TOKENS, BLOCK_SIZE, MASK_TOKEN, TORCH_DTYPE."""
    datasets = os.environ.get("DATASETS", "DocVQA_VAL").split()
    dataset_class = os.environ.get("DATASET_CLASS", "ImageVQADataset")

    proc = os.environ.get("PROCESSOR_PATH", "").strip()
    model_cfg: Dict[str, Any] = {
        "class": "Fast_dVLM",
        "model_path": os.environ["MODEL_PATH_ABS"],
        "processor_path": proc or os.environ["MODEL_PATH_ABS"],
        "torch_dtype": os.environ.get("TORCH_DTYPE", "bfloat16"),
        "max_tokens": int(os.environ.get("MAX_TOKENS", "2048")),
        "mask_token": os.environ.get("MASK_TOKEN", "|<MASK>|"),
    }
    bs = os.environ.get("BLOCK_SIZE", "").strip()
    if bs:
        model_cfg["block_size"] = int(bs)

    cfg: Dict[str, Any] = {"model": {"Fast_dVLM": model_cfg}, "data": {}}
    for name in datasets:
        cfg["data"][name] = {"class": dataset_class, "dataset": name}
        print(f"[config] {name} -> {dataset_class}")

    cfg_path = os.environ["CFG_PATH"]
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] Saved to: {cfg_path}")


class Fast_dVLM:
    """VLMEval wrapper: loads Fast-dVLM (or any trust_remote_code) checkpoint and calls its ``generate``."""

    def __init__(
        self,
        model_path: str,
        processor_path: Optional[str] = None,
        torch_dtype: Union[str, Any] = "bfloat16",
        max_tokens: int = 2048,
        block_size: Optional[int] = None,
        mask_token: str = "|<MASK>|",
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        _ = kwargs  # VLMEval may pass extra keys from template configs
        self.processor_path = processor_path or model_path
        self.max_tokens = int(max_tokens)
        self._block_size = int(block_size) if block_size is not None else None
        self.mask_token = mask_token

        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        td: Any = torch_dtype
        if isinstance(td, str):
            if td == "auto":
                td = "auto"
            elif hasattr(torch, td):
                td = getattr(torch, td)
            else:
                td = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=td,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(self.processor_path, use_fast=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor.tokenizer = self.tokenizer

        self._eos = int(
            getattr(self.model.config, "eos_token_id", None)
            or self.tokenizer.eos_token_id
            or 151645
        )

    def _build_user_messages(self, image: Optional[str], prompt: str) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if image:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _generate_one(self, prompt: str, image: Optional[str]) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = self._build_user_messages(image, prompt)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda:0")

        mask_id = int(self.tokenizer.encode(self.mask_token)[0])
        block = self._block_size
        if block is None:
            block = int(getattr(self.model.config, "bd_size", 32))

        gen_kw: Dict[str, Any] = {
            "input_ids": inputs.input_ids,
            "tokenizer": self.processor.tokenizer,
            "block_size": block,
            "max_tokens": self.max_tokens,
            "mask_id": mask_id,
            "stop_token": self._eos,
        }
        if hasattr(inputs, "pixel_values"):
            gen_kw["pixel_values"] = inputs.pixel_values
        if hasattr(inputs, "image_grid_thw"):
            gen_kw["image_grid_thw"] = inputs.image_grid_thw

        with torch.inference_mode():
            generated_ids = self.model.generate(**gen_kw)

        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return out[0] if out else ""

    def generate(
        self,
        message=None,
        dataset=None,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        _ = dataset, kwargs
        if message is not None:
            if isinstance(message, list):
                image_path = None
                prompt_parts: List[str] = []
                for item in message:
                    if isinstance(item, dict):
                        if item.get("type") == "image" and "value" in item:
                            image_path = item["value"]
                        elif item.get("type") == "text" and "value" in item:
                            prompt_parts.append(item["value"])
                if prompt_parts:
                    prompt = " ".join(prompt_parts)
                if image_path:
                    image = image_path
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")

        if prompt is None:
            raise ValueError("No prompt found in message or prompt parameter")

        if isinstance(prompt, list):
            images = image if isinstance(image, list) else [image] * len(prompt) if image else [None] * len(prompt)
            return [self._generate_one(p, img) for p, img in zip(prompt, images)]

        img0 = image if isinstance(image, str) else (image[0] if image else None)
        return self._generate_one(prompt, img0)

    def set_dump_image(self, dump_image: bool) -> None:
        self.dump_image = dump_image


try:
    import vlmeval.api as api

    if not hasattr(api, "Fast_dVLM"):
        api.Fast_dVLM = Fast_dVLM
except ImportError:
    pass

try:
    import vlmeval.vlm as vlm

    if not hasattr(vlm, "Fast_dVLM"):
        vlm.Fast_dVLM = Fast_dVLM
except ImportError:
    pass


def _launch_vlmeval() -> None:
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    if LOCAL_WORLD_SIZE > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(LOCAL_RANK)
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        print(
            f"[Fast_dVLM] RANK={RANK}/{WORLD_SIZE}, LOCAL_RANK={LOCAL_RANK}, "
            f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
        )

    # Script dir is on sys.path[0], not cwd; VLMEvalKit ``run.py`` lives next to ``vlmeval/``.
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _vlm_root = os.path.join(_repo_root, "third_party", "VLMEvalKit")
    if os.path.isfile(os.path.join(_vlm_root, "run.py")) and _vlm_root not in sys.path:
        sys.path.insert(0, _vlm_root)

    import run  # noqa: E402  # VLMEvalKit run.py

    print(f"[Fast_dVLM] sys.argv: {sys.argv}")
    run.main()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "write-config":
        write_vlmeval_config()
    else:
        _launch_vlmeval()
