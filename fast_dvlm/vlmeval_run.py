#!/usr/bin/env python3
"""VLMEvalKit: write JSON config + register ``Fast_dVLM`` + delegate to VLMEvalKit ``run.py``.

  python vlmeval_run.py write-config   # env: CFG_PATH, MODEL_PATH_ABS, DATASETS, …
  torchrun … vlmeval_run.py --config … --work-dir …

Two inference backends, selected by ``BACKEND`` (config key ``backend``):

* ``hf``     — checkpoint ``modeling.py`` ``generate`` via ``AutoModelForCausalLM``
               (trust_remote_code). Same stack as ``run_chatbot.py``. Default.
* ``sglang`` — the vendored SGLang fork (``third_party/sglang``) ``sgl.Engine``,
               same stack as ``run_chatbot_sglang.py``. Honors ``ALGORITHM``
               (mdm = HierarchyBlock, spec = SpeculativeBlock) and
               ``QUANTIZATION`` (e.g. ``w8a8_fp8``; requires SM89+).
"""
from __future__ import annotations

import atexit
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union

# sglang algorithm -> dLLM decoding class (mirrors run_chatbot_sglang.py).
ALGO_MAP = {
    "mdm": "HierarchyBlock",
    "spec": "SpeculativeBlock",
}


def write_vlmeval_config() -> None:
    """Env: CFG_PATH, MODEL_PATH_ABS; optional DATASETS, DATASET_CLASS,
    PROCESSOR_PATH (defaults to MODEL_PATH_ABS), MAX_TOKENS, BLOCK_SIZE,
    MASK_TOKEN, TORCH_DTYPE, BACKEND (hf|sglang), ALGORITHM (mdm|spec),
    QUANTIZATION (e.g. w8a8_fp8), MEM_FRACTION_STATIC."""
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

    backend = os.environ.get("BACKEND", "hf").strip().lower() or "hf"
    model_cfg["backend"] = backend
    if backend == "sglang":
        model_cfg["algorithm"] = os.environ.get("ALGORITHM", "mdm").strip().lower() or "mdm"
        model_cfg["mem_fraction_static"] = float(os.environ.get("MEM_FRACTION_STATIC", "0.75"))
        quant = os.environ.get("QUANTIZATION", "").strip()
        if quant:
            model_cfg["quantization"] = quant

    cfg: Dict[str, Any] = {"model": {"Fast_dVLM": model_cfg}, "data": {}}
    for name in datasets:
        cfg["data"][name] = {"class": dataset_class, "dataset": name}
        print(f"[config] {name} -> {dataset_class}")

    cfg_path = os.environ["CFG_PATH"]
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] Saved to: {cfg_path} (backend={backend})")


class Fast_dVLM:
    """VLMEval wrapper for Fast-dVLM. Backend ``hf`` (checkpoint ``generate``) or
    ``sglang`` (vendored SGLang fork ``sgl.Engine``)."""

    def __init__(
        self,
        model_path: str,
        processor_path: Optional[str] = None,
        torch_dtype: Union[str, Any] = "bfloat16",
        max_tokens: int = 2048,
        block_size: Optional[int] = None,
        mask_token: str = "|<MASK>|",
        backend: str = "hf",
        algorithm: str = "mdm",
        quantization: Optional[str] = None,
        mem_fraction_static: float = 0.75,
        **kwargs: Any,
    ) -> None:
        _ = kwargs  # VLMEval may pass extra keys from template configs
        self.backend = (backend or "hf").strip().lower()
        self.processor_path = processor_path or model_path
        self.model_path = model_path
        self.max_tokens = int(max_tokens)
        self._block_size = int(block_size) if block_size is not None else None
        self.mask_token = mask_token
        self.algorithm = (algorithm or "mdm").strip().lower()
        self.quantization = quantization or None
        self.mem_fraction_static = float(mem_fraction_static)

        if self.backend == "sglang":
            self._init_sglang(torch_dtype)
        else:
            self._init_hf(torch_dtype)

    # --- HF backend (unchanged behavior) ---------------------------------
    def _init_hf(self, torch_dtype: Union[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

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
            self.model_path,
            torch_dtype=td,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(self.processor_path, use_fast=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.processor.tokenizer = self.tokenizer

        self._eos = int(
            getattr(self.model.config, "eos_token_id", None)
            or self.tokenizer.eos_token_id
            or 151645
        )

    # --- SGLang backend (mirrors run_chatbot_sglang.py) ------------------
    def _init_sglang(self, torch_dtype: Union[str, Any]) -> None:
        from transformers import AutoProcessor, AutoTokenizer

        # `import sglang` must resolve to the pip-installed fork
        # (third_party/sglang); drop this script's dir so a stray local
        # `sglang/` next to it can't shadow the package.
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path[:] = [p for p in sys.path if os.path.abspath(p) != here]
        os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")

        import sglang as sgl

        self.processor = AutoProcessor.from_pretrained(self.processor_path, use_fast=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor.tokenizer = self.tokenizer

        if self.algorithm not in ALGO_MAP:
            raise ValueError(
                f"Unknown sglang algorithm {self.algorithm!r}; expected one of {list(ALGO_MAP)}"
            )
        dllm_algo = ALGO_MAP[self.algorithm]

        dtype = torch_dtype if isinstance(torch_dtype, str) and torch_dtype else "bfloat16"
        if dtype not in ("bfloat16", "float16", "half", "auto"):
            dtype = "bfloat16"

        engine_kwargs: Dict[str, Any] = dict(
            model_path=self.model_path,
            trust_remote_code=True,
            dtype=dtype,
            mem_fraction_static=self.mem_fraction_static,
            max_running_requests=1,
            chunked_prefill_size=16384,
            dllm_algorithm=dllm_algo,
            disable_cuda_graph=False,
            log_level="warning",
            enable_metrics=True,
            mm_attention_backend="triton_attn",
        )
        if self.quantization:
            engine_kwargs["quantization"] = self.quantization

        print(
            f"[Fast_dVLM] Launching sglang Engine dllm_algorithm={dllm_algo}"
            f"{f', quantization={self.quantization}' if self.quantization else ''} ..."
        )
        # run_eval.sh launches us under `torchrun` for data-parallel sharding
        # (each rank = one independent worker). torchrun injects RANK/
        # WORLD_SIZE/MASTER_PORT/TORCHELASTIC_* into the env. sgl.Engine forks
        # a scheduler subprocess that, seeing those vars, tries to join a
        # torch.distributed group / TCPStore nobody serves -> 600s TCPStore
        # timeout and hang. Each rank must run an *independent* single-GPU
        # engine, so scrub those vars only across Engine construction, then
        # restore them so VLMEvalKit's RANK/WORLD_SIZE dataset sharding and
        # result aggregation still work afterwards.
        _DIST_ENV_KEYS = (
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
            "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
            "ROLE_NAME", "MASTER_ADDR", "MASTER_PORT",
            "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_USE_AGENT_STORE",
            "TORCHELASTIC_ERROR_FILE", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "PET_NPROC_PER_NODE",
        )
        _saved_env = {k: os.environ.pop(k) for k in _DIST_ENV_KEYS if k in os.environ}
        try:
            self.engine = sgl.Engine(**engine_kwargs)
        finally:
            os.environ.update(_saved_env)
        atexit.register(self._shutdown_engine)

    def _shutdown_engine(self) -> None:
        eng = getattr(self, "engine", None)
        if eng is not None:
            try:
                eng.shutdown()
            except Exception:
                pass
            self.engine = None

    def _build_user_messages(self, image: Optional[str], prompt: str) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if image:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _build_input_ids(self, prompt: str, image: Optional[str]) -> List[int]:
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
        )
        return inputs.input_ids[0].tolist()

    def _generate_one_sglang(self, prompt: str, image: Optional[str]) -> str:
        input_ids = self._build_input_ids(prompt, image)
        out = self.engine.generate(
            input_ids=input_ids,
            image_data=[image] if image else None,
            sampling_params={"max_new_tokens": self.max_tokens, "temperature": 0.0},
        )
        if isinstance(out, list):
            out = out[0]
        return out["text"]

    def _generate_one_hf(self, prompt: str, image: Optional[str]) -> str:
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

    def _generate_one(self, prompt: str, image: Optional[str]) -> str:
        if self.backend == "sglang":
            return self._generate_one_sglang(prompt, image)
        return self._generate_one_hf(prompt, image)

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
