import logging
import copy
import os
import random
import sys
import time
import traceback
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional

_DETERMINISTIC_CUDA_VALUE = os.getenv("DETERMINISTIC_CUDA", "1").strip().lower()
if _DETERMINISTIC_CUDA_VALUE not in {
    "0",
    "1",
    "false",
    "true",
    "no",
    "yes",
    "off",
    "on",
}:
    raise RuntimeError(
        "DETERMINISTIC_CUDA must be a boolean value; "
        f"got {_DETERMINISTIC_CUDA_VALUE!r}"
    )
DETERMINISTIC_CUDA = _DETERMINISTIC_CUDA_VALUE in {"1", "true", "yes", "on"}

# cuBLAS reads this before the first CUDA context is initialized, so configure
# it before importing torch. The launcher also exports it for clarity.
if DETERMINISTIC_CUDA:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
LLADA_DIR = ROOT / "v1" / "llada"
sys.path.insert(0, str(LLADA_DIR))

from generate import generate, generate_with_dual_cache, generate_first_block, generate_with_dual_cache_first_block  # noqa: E402
from model.modeling_llada import LLaDAModelLM  # noqa: E402
from reverse_loglikelihood import get_reverse_log_likelihood
from forward_loglikelihood import get_forward_log_likelihood
from joint_loglikelihood import get_joint_log_likelihood

MODEL_API_NAME = "Llada"
MODEL_HF_NAME = "GSAI-ML/LLaDA-8B-Instruct"
LOG_LIKELIHOOD_CFG_SCALE = 0.0
LOG_LIKELIHOOD_MASK_ID = 126336

ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
START_HEADER = "<|startoftext|>"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("serve_agentboard_llada")

app = FastAPI(title="AgentBoard LLaDA Server")

_model = None
_tokenizer = None
_device = None
_load_lock = Lock()


def _set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_deterministic_cuda() -> None:
    if not DETERMINISTIC_CUDA:
        return

    # Strict mode raises instead of silently executing a known
    # nondeterministic CUDA operation.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is not None:
        matmul_backend = getattr(cuda_backends, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = False

        # LLaDA enables Flash SDPA while constructing the model. Math SDPA is
        # slower but is the backend suitable for strict reproducibility.
        if hasattr(cuda_backends, "enable_flash_sdp"):
            cuda_backends.enable_flash_sdp(False)
        if hasattr(cuda_backends, "enable_mem_efficient_sdp"):
            cuda_backends.enable_mem_efficient_sdp(False)
        if hasattr(cuda_backends, "enable_math_sdp"):
            cuda_backends.enable_math_sdp(True)


SERVER_SEED = int(os.getenv("SEED", "42"))
_configure_deterministic_cuda()
_set_global_seed(SERVER_SEED)
logger.info("Global seed set to: %d", SERVER_SEED)
logger.info(
    "Deterministic CUDA: enabled=%s CUBLAS_WORKSPACE_CONFIG=%s "
    "NVIDIA_TF32_OVERRIDE=%s TORCH_COMPILE_DISABLE=%s",
    DETERMINISTIC_CUDA,
    os.getenv("CUBLAS_WORKSPACE_CONFIG"),
    os.getenv("NVIDIA_TF32_OVERRIDE"),
    os.getenv("TORCH_COMPILE_DISABLE"),
)


class Message(BaseModel):
    role: str
    content: str


class TokenRequest(BaseModel):
    model: str
    messages: List[Message]


class GenerateRequest(TokenRequest):
    gen_length: int = 128
    temperature: float = 0.0
    steps: int = 128
    dual_cache: bool = True
    block_size: int = 32
    threshold: Optional[float] = 0.9
    return_tokens: bool = True

class GenerateMultiRequest(GenerateRequest):
    block_generation: str = ""

class LikelihoodRequest(BaseModel):
    model: str
    reflect_list: List[Any]
    goal: str = ""
    mc_num: int = 128
    batch_size: int = 16

def _validate_model(model: str) -> None:
    if model != MODEL_API_NAME:
        raise HTTPException(
            status_code=400,
            detail=f'Unsupported model "{model}". Only "{MODEL_API_NAME}" is supported.',
        )


def _messages_to_dicts(messages: List[Message]) -> List[dict]:
    return [
        message.model_dump() if hasattr(message, "model_dump") else message.dict()
        for message in messages
    ]


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with _load_lock:
            if _tokenizer is None:
                logger.info("Loading tokenizer: %s", MODEL_HF_NAME)
                _tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_HF_NAME,
                    trust_remote_code=True,
                )
                logger.info("Tokenizer loaded")
    return _tokenizer


def _get_model_and_tokenizer():
    global _device, _model, _tokenizer
    if _model is None:
        with _load_lock:
            if _model is None:
                start_time = time.perf_counter()
                _device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(
                    "Loading model: %s device=%s dtype=%s",
                    MODEL_HF_NAME,
                    _device,
                    torch.bfloat16,
                )
                if _tokenizer is None:
                    _tokenizer = AutoTokenizer.from_pretrained(
                        MODEL_HF_NAME,
                        trust_remote_code=True,
                    )
                _model = (
                    LLaDAModelLM.from_pretrained(
                        MODEL_HF_NAME,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                    )
                    .to(_device)
                    .eval()
                )
                # LLaDAModelLM construction enables Flash SDPA internally.
                # Restore the deterministic backend before the first forward.
                _configure_deterministic_cuda()
                logger.info("Model loaded in %.2fs", time.perf_counter() - start_time)
    return _model, _tokenizer, _device


def _encode_messages(tokenizer, messages: List[Message]) -> torch.Tensor:
    chat_input = tokenizer.apply_chat_template(
        _messages_to_dicts(messages),
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer(chat_input)["input_ids"]
    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)


def _generated_token_count(generated_ids: torch.Tensor, tokenizer) -> int:
    special_ids = set(tokenizer.all_special_ids or [])
    return sum(1 for token_id in generated_ids.tolist() if token_id not in special_ids)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_API_NAME}


@app.get("/models")
def models():
    return {"models": [MODEL_API_NAME]}


@app.post("/tokens")
def tokens(request: TokenRequest, authorization: Optional[str] = Header(default=None)):
    _validate_model(request.model)
    del authorization

    tokenizer = _get_tokenizer()
    input_ids = _encode_messages(tokenizer, request.messages)
    num_tokens = int(input_ids.shape[1])
    logger.info("Token count: model=%s tokens=%d", request.model, num_tokens)
    return {"num_of_tokens": num_tokens}


@app.post("/generate")
def generate_response(
    request: GenerateRequest,
    authorization: Optional[str] = Header(default=None),
):
    _validate_model(request.model)
    del authorization

    try:
        model, tokenizer, device = _get_model_and_tokenizer()

        prompt = _encode_messages(tokenizer, request.messages).to(device)

        start_time = time.perf_counter()
        with torch.inference_mode():
            if request.dual_cache:
                output_ids, _ = generate_with_dual_cache(
                    model,
                    prompt,
                    steps=request.steps,
                    gen_length=request.gen_length,
                    block_length=request.block_size,
                    temperature=request.temperature,
                    remasking="low_confidence",
                    threshold=request.threshold,
                )
            else:
                output_ids, _ = generate(
                    model,
                    prompt,
                    steps=request.steps,
                    gen_length=request.gen_length,
                    block_length=request.block_size,
                    temperature=request.temperature,
                    remasking="low_confidence",
                    threshold=request.threshold,
                )

        generated_ids = output_ids[0, prompt.shape[1] :].detach().cpu()
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = _generated_token_count(generated_ids, tokenizer)
        latency = time.perf_counter() - start_time
        tokens_per_second = generated_tokens / latency if latency > 0 else 0.0

        logger.info(
            "Generation complete: model=%s prompt_tokens=%d generated_tokens=%d "
            "latency=%.3fs tokens_per_sec=%.2f dual_cache=%s block_size=%d "
            "steps=%d gen_length=%d threshold=%s",
            request.model,
            int(prompt.shape[1]),
            generated_tokens,
            latency,
            tokens_per_second,
            request.dual_cache,
            request.block_size,
            request.steps,
            request.gen_length,
            request.threshold,
        )

        return {"response": response, "token": generated_tokens}
    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Generation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "traceback": tb},
        ) from exc

@app.post("/generate_first_block")
def generate_response_first_block(
    request: GenerateMultiRequest,
    authorization: Optional[str] = Header(default=None),
):
    _validate_model(request.model)
    del authorization

    try:
        model, tokenizer, device = _get_model_and_tokenizer()

        prompt = _encode_messages(tokenizer, request.messages).to(device)

        if len(request.block_generation) > 0:
            block_generation_ids = tokenizer(request.block_generation, add_special_tokens=False)["input_ids"]
            block_generation_ids = torch.tensor(block_generation_ids, dtype=torch.long, device=device).unsqueeze(0)
            prompt = torch.concat([prompt, block_generation_ids], dim=-1)

        start_time = time.perf_counter()
        with torch.inference_mode():
            if request.dual_cache:
                output_ids, _ = generate_with_dual_cache_first_block(
                    model,
                    prompt,
                    steps=request.steps,
                    gen_length=request.gen_length,
                    block_length=request.block_size,
                    temperature=request.temperature,
                    remasking="low_confidence",
                    threshold=request.threshold,
                )
            else:
                output_ids, _ = generate_first_block(
                    model,
                    prompt,
                    steps=request.steps,
                    gen_length=request.gen_length,
                    block_length=request.block_size,
                    temperature=request.temperature,
                    remasking="low_confidence",
                    threshold=request.threshold,
                )

        generated_ids = output_ids[0].detach().cpu()
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = _generated_token_count(generated_ids, tokenizer)
        latency = time.perf_counter() - start_time
        tokens_per_second = generated_tokens / latency if latency > 0 else 0.0

        logger.info(
            "Generation complete: model=%s prompt_tokens=%d generated_tokens=%d "
            "latency=%.3fs tokens_per_sec=%.2f dual_cache=%s block_size=%d "
            "steps=%d gen_length=%d threshold=%s",
            request.model,
            int(prompt.shape[1]),
            generated_tokens,
            latency,
            tokens_per_second,
            request.dual_cache,
            request.block_size,
            request.steps,
            request.gen_length,
            request.threshold,
        )

        return {"response": response, "token": generated_tokens}
    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Generation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "traceback": tb},
        ) from exc

@app.post("/reverse_loglikelihood")
def reverse_loglikelihood_response(
    request: LikelihoodRequest,
    authorization: Optional[str] = Header(default=None),
):
    _validate_model(request.model)
    del authorization

    if not request.goal:
        raise HTTPException(status_code=400, detail="goal must be provided")
    if not request.reflect_list:
        raise HTTPException(status_code=400, detail="reflect_list must contain at least one candidate")
    if request.mc_num <= 0 or request.batch_size <= 0:
        raise HTTPException(status_code=400, detail="mc_num and batch_size must be positive")
    if request.mc_num % request.batch_size != 0:
        raise HTTPException(status_code=400, detail="mc_num must be divisible by batch_size")

    try:
        model, tokenizer, device = _get_model_and_tokenizer()

        all_scores = []
        start_time = time.perf_counter()
        goal_tokens = tokenizer(
            request.goal,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        target_len = int(goal_tokens.shape[0])
        if target_len == 0:
            raise HTTPException(status_code=400, detail="goal cannot tokenize to an empty sequence")

        with torch.inference_mode():
            for reflect_candidate in request.reflect_list:
                prompt_template = tokenizer.apply_chat_template(
                    reflect_candidate,
                    add_generation_prompt=False,
                    tokenize=False,
                )
                if prompt_template.endswith(ASSISTANT_HEADER):
                    prompt_template = prompt_template[:-len(ASSISTANT_HEADER)]

                prompt = tokenizer(
                    prompt_template,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0]

                k = [
                    torch.randint(1, target_len + 1, (), device=device)
                    for _ in range(request.mc_num // request.batch_size)
                ]
                shuffled_seq = [
                    torch.stack(
                        [
                            torch.randperm(target_len, device=device)
                            for _ in range(request.batch_size)
                        ],
                        dim=0,
                    )
                    for _ in range(request.mc_num // request.batch_size)
                ]

                seq = torch.concatenate([goal_tokens, prompt])[None, :]
                seq = seq.repeat((request.batch_size, 1)).to(device)
                prompt_index = torch.arange(seq.shape[1], device=device) >= target_len
                score = get_reverse_log_likelihood(
                    model,
                    seq,
                    prompt_index,
                    shuffled_seq,
                    k,
                    mc_num=request.mc_num,
                    batch_size=request.batch_size,
                    cfg_scale=LOG_LIKELIHOOD_CFG_SCALE,
                    mask_id=LOG_LIKELIHOOD_MASK_ID,
                )
                all_scores.append(float(score))

        normalized_scores = torch.softmax(
            torch.tensor(all_scores, dtype=torch.float64),
            dim=0,
        ).detach().cpu().tolist()
    
        logger.info(
            "Loglikelihood complete: candidates=%d mc_num=%d batch_size=%d latency=%.3fs",
            len(request.reflect_list),
            request.mc_num,
            request.batch_size,
            time.perf_counter() - start_time,
        )

        return {"normalized_loglikelihoods": normalized_scores}
    
    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Loglikelihood failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "traceback": tb},
        ) from exc
    
@app.post("/forward_loglikelihood")
def forward_loglikelihood_response(
    request: LikelihoodRequest,
    authorization: Optional[str] = Header(default=None),
):
    _validate_model(request.model)
    del authorization

    if not request.goal:
        raise HTTPException(status_code=400, detail="goal must be provided")
    if not request.reflect_list:
        raise HTTPException(status_code=400, detail="reflect_list must contain at least one candidate")
    if request.mc_num <= 0 or request.batch_size <= 0:
        raise HTTPException(status_code=400, detail="mc_num and batch_size must be positive")
    if request.mc_num % request.batch_size != 0:
        raise HTTPException(status_code=400, detail="mc_num must be divisible by batch_size")

    try:
        model, tokenizer, device = _get_model_and_tokenizer()

        all_scores = []
        start_time = time.perf_counter()

        if len(request.reflect_list[0]) > 1:
            first_reflect = copy.deepcopy(request.reflect_list[0])
            first_reflect[0]["content"] = (
                f"{request.goal}\n{first_reflect[0]['content']}"
            )
            prompt_template = tokenizer.apply_chat_template(
                first_reflect[:-1],
                add_generation_prompt=False,
                tokenize=False,
            )
        else:
            prompt_template = copy.deepcopy(request.goal)

        if prompt_template.endswith(ASSISTANT_HEADER):
            prompt_template = prompt_template[: -len(ASSISTANT_HEADER)]

        prompt = tokenizer(
            prompt_template,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]

        prompt_len = int(prompt.shape[0])
        if prompt_len == 0:
            raise HTTPException(status_code=400, detail="goal cannot tokenize to an empty sequence")

        with torch.inference_mode():
            for reflect_candidate in request.reflect_list:
                act_candidate_template = tokenizer.apply_chat_template(
                    reflect_candidate[-1:],
                    add_generation_prompt=False,
                    tokenize=False,
                )
                act_candidate_template = act_candidate_template.replace(START_HEADER, "")
                if act_candidate_template.endswith(ASSISTANT_HEADER):
                    act_candidate_template = act_candidate_template[: -len(ASSISTANT_HEADER)]

                act_candidate = tokenizer(
                    act_candidate_template,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0]

                target_len = int(act_candidate.shape[0])
                if target_len == 0:
                    raise HTTPException(status_code=400, detail="goal cannot tokenize to an empty sequence")

                k = [
                    torch.randint(1, target_len + 1, (), device=device)
                    for _ in range(request.mc_num // request.batch_size)
                ]
                shuffled_seq = [
                    torch.stack(
                        [
                            torch.randperm(target_len, device=device)
                            for _ in range(request.batch_size)
                        ],
                        dim=0,
                    )
                    for _ in range(request.mc_num // request.batch_size)
                ]

                seq = torch.concatenate([prompt, act_candidate])[None, :]
                seq = seq.repeat((request.batch_size, 1)).to(device)
                prompt_index = torch.arange(seq.shape[1], device=device) < prompt_len

                score = get_forward_log_likelihood(
                    model,
                    seq,
                    prompt_index,
                    shuffled_seq,
                    k,
                    mc_num=request.mc_num,
                    batch_size=request.batch_size,
                    cfg_scale=LOG_LIKELIHOOD_CFG_SCALE,
                    mask_id=LOG_LIKELIHOOD_MASK_ID,
                )
                all_scores.append(float(score))

        normalized_scores = torch.softmax(
            torch.tensor(all_scores, dtype=torch.float64),
            dim=0,
        ).detach().cpu().tolist()

        logger.info(
            "Loglikelihood complete: candidates=%d mc_num=%d batch_size=%d latency=%.3fs",
            len(request.reflect_list),
            request.mc_num,
            request.batch_size,
            time.perf_counter() - start_time,
        )

        return {"normalized_loglikelihoods": normalized_scores}

    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Loglikelihood failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "traceback": tb},
        ) from exc
    
@app.post("/multi_loglikelihood")
def multi_loglikelihood_response(
    request: LikelihoodRequest,
    authorization: Optional[str] = Header(default=None),
):
    _validate_model(request.model)
    del authorization

    if not request.goal:
        raise HTTPException(status_code=400, detail="goal must be provided")
    if not request.reflect_list:
        raise HTTPException(status_code=400, detail="reflect_list must contain at least one candidate")
    if request.mc_num <= 0 or request.batch_size <= 0:
        raise HTTPException(status_code=400, detail="mc_num and batch_size must be positive")
    if request.mc_num % request.batch_size != 0:
        raise HTTPException(status_code=400, detail="mc_num must be divisible by batch_size")

    try:
        model, tokenizer, device = _get_model_and_tokenizer()

        reverse_all_scores = []
        forward_all_scores = []

        start_time = time.perf_counter()

        ### for reverse inference
        goal_tokens = tokenizer(
            request.goal,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        reverse_target_len = int(goal_tokens.shape[0])
        if reverse_target_len == 0:
            raise HTTPException(status_code=400, detail="goal cannot tokenize to an empty sequence")

        ### for forward inference
        if len(request.reflect_list[0]) > 1:
            first_reflect = copy.deepcopy(request.reflect_list[0])
            first_reflect[0]["content"] = (
                f"{request.goal}\n{first_reflect[0]['content']}"
            )
            forward_prompt_template = tokenizer.apply_chat_template(
                first_reflect[:-1],
                add_generation_prompt=False,
                tokenize=False,
            )
        else:
            forward_prompt_template = copy.deepcopy(request.goal)

        if forward_prompt_template.endswith(ASSISTANT_HEADER):
            forward_prompt_template = forward_prompt_template[: -len(ASSISTANT_HEADER)]

        forward_prompt = tokenizer(
            forward_prompt_template,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        
        forward_prompt_len = int(forward_prompt.shape[0])
        if forward_prompt_len == 0:
            raise HTTPException(status_code=400, detail="goal cannot tokenize to an empty sequence")
        
        with torch.inference_mode():
            for reflect_candidate in request.reflect_list:
                ### reverse
                reverse_prompt_template = tokenizer.apply_chat_template(
                    reflect_candidate,
                    add_generation_prompt=False,
                    tokenize=False,
                )
                if reverse_prompt_template.endswith(ASSISTANT_HEADER):
                    reverse_prompt_template = reverse_prompt_template[: -len(ASSISTANT_HEADER)]

                reverse_prompt = tokenizer(
                    reverse_prompt_template,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0]

                reverse_k = [
                    torch.randint(1, reverse_target_len + 1, (), device=device)
                    for _ in range(request.mc_num // request.batch_size)
                ]
                reverse_shuffled_seq = [
                    torch.stack(
                        [
                            torch.randperm(reverse_target_len, device=device)
                            for _ in range(request.batch_size)
                        ],
                        dim=0,
                    )
                    for _ in range(request.mc_num // request.batch_size)
                ]

                reverse_seq = torch.concatenate([goal_tokens, reverse_prompt])[None, :]
                reverse_seq = reverse_seq.repeat((request.batch_size, 1)).to(device)
                reverse_prompt_index = torch.arange(reverse_seq.shape[1], device=device) >= reverse_target_len
                reverse_score = get_reverse_log_likelihood(
                    model,
                    reverse_seq,
                    reverse_prompt_index,
                    reverse_shuffled_seq,
                    reverse_k,
                    mc_num=request.mc_num,
                    batch_size=request.batch_size,
                    cfg_scale=LOG_LIKELIHOOD_CFG_SCALE,
                    mask_id=LOG_LIKELIHOOD_MASK_ID,
                )
                reverse_all_scores.append(float(reverse_score))

                # forward
                act_candidate_template = tokenizer.apply_chat_template(
                    reflect_candidate[-1:],
                    add_generation_prompt=False,
                    tokenize=False,
                )
                act_candidate_template = act_candidate_template.replace(START_HEADER, "")
                if act_candidate_template.endswith(ASSISTANT_HEADER):
                    act_candidate_template = act_candidate_template[: -len(ASSISTANT_HEADER)]

                act_candidate = tokenizer(
                    act_candidate_template,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0]

                forward_target_len = int(act_candidate.shape[0])
                if forward_target_len == 0:
                    raise HTTPException(status_code=400, detail="goal cannot tokenize to an empty sequence")

                forward_k = [
                    torch.randint(1, forward_target_len + 1, (), device=device)
                    for _ in range(request.mc_num // request.batch_size)
                ]
                forward_shuffled_seq = [
                    torch.stack(
                        [
                            torch.randperm(forward_target_len, device=device)
                            for _ in range(request.batch_size)
                        ],
                        dim=0,
                    )
                    for _ in range(request.mc_num // request.batch_size)
                ]

                forward_seq = torch.concatenate([forward_prompt, act_candidate])[None, :]
                forward_seq = forward_seq.repeat((request.batch_size, 1)).to(device)
                forward_prompt_index = torch.arange(forward_seq.shape[1], device=device) < forward_prompt_len

                forward_score = get_forward_log_likelihood(
                    model,
                    forward_seq,
                    forward_prompt_index,
                    forward_shuffled_seq,
                    forward_k,
                    mc_num=request.mc_num,
                    batch_size=request.batch_size,
                    cfg_scale=LOG_LIKELIHOOD_CFG_SCALE,
                    mask_id=LOG_LIKELIHOOD_MASK_ID,
                )
                forward_all_scores.append(float(forward_score))
                
        reverse_score_tensor = torch.tensor(reverse_all_scores, dtype=torch.float64)
        forward_score_tensor = torch.tensor(forward_all_scores, dtype=torch.float64)
        reverse_normalized_scores = torch.softmax(reverse_score_tensor, dim=0)
        forward_normalized_scores = torch.softmax(forward_score_tensor, dim=0)

        combined_scores = reverse_normalized_scores * forward_normalized_scores
        combined_normalizer = combined_scores.sum()
        if (
            not torch.isfinite(combined_normalizer).item()
            or combined_normalizer.item() <= 0
        ):
            normalized_scores_tensor = torch.softmax(
                reverse_score_tensor + forward_score_tensor,
                dim=0,
            )
        else:
            normalized_scores_tensor = combined_scores / combined_normalizer
        normalized_scores = normalized_scores_tensor.detach().cpu().tolist()
        
        logger.info(
            "Loglikelihood complete: candidates=%d mc_num=%d batch_size=%d latency=%.3fs",
            len(request.reflect_list),
            request.mc_num,
            request.batch_size,
            time.perf_counter() - start_time,
        )

        return {"normalized_loglikelihoods": normalized_scores}
    
    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Loglikelihood failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "traceback": tb},
        ) from exc

@app.post("/joint_loglikelihood")
def joint_loglikelihood_response(
    request: LikelihoodRequest,
    authorization: Optional[str] = Header(default=None),
):
    _validate_model(request.model)
    del authorization

    if not request.goal:
        raise HTTPException(status_code=400, detail="goal must be provided")
    if not request.reflect_list:
        raise HTTPException(status_code=400, detail="reflect_list must contain at least one candidate")
    if request.mc_num <= 0 or request.batch_size <= 0:
        raise HTTPException(status_code=400, detail="mc_num and batch_size must be positive")
    if request.mc_num % request.batch_size != 0:
        raise HTTPException(status_code=400, detail="mc_num must be divisible by batch_size")

    try:
        model, tokenizer, device = _get_model_and_tokenizer()

        all_scores = []
        start_time = time.perf_counter()
        goal_tokens = tokenizer(
            request.goal,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        goal_len = int(goal_tokens.shape[0])
        if goal_len == 0:
            raise HTTPException(status_code=400, detail="goal cannot tokenize to an empty sequence")

        prompt_list = copy.deepcopy(request.reflect_list[0])
        prompt_template = tokenizer.apply_chat_template(
            prompt_list[:-1],
            add_generation_prompt=False,
            tokenize=False,
        )
        if prompt_template.endswith(ASSISTANT_HEADER):
            prompt_template = prompt_template[: -len(ASSISTANT_HEADER)]

        prompt_tokens = tokenizer(
            prompt_template,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        
        prompt_len = int(prompt_tokens.shape[0])
        if prompt_len == 0:
            raise HTTPException(status_code=400, detail="goal cannot tokenize to an empty sequence")
        
        with torch.inference_mode():
            for reflect_candidate in request.reflect_list:
                act_candidate_template = tokenizer.apply_chat_template(
                    reflect_candidate[-1:],
                    add_generation_prompt=False,
                    tokenize=False,
                )
                act_candidate_template = act_candidate_template.replace(START_HEADER, "")
                if act_candidate_template.endswith(ASSISTANT_HEADER):
                    act_candidate_template = act_candidate_template[:-len(ASSISTANT_HEADER)]

                act_candidate_tokens = tokenizer(
                    act_candidate_template,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0]

                act_candidate_len = int(act_candidate_tokens.shape[0])
                if act_candidate_len == 0:
                    raise HTTPException(status_code=400, detail="goal cannot tokenize to an empty sequence")
        
                k = [
                    torch.randint(1, goal_len + act_candidate_len + 1, (), device=device)
                    for _ in range(request.mc_num // request.batch_size)
                ]
                shuffled_seq = [
                    torch.stack(
                        [
                            torch.randperm(goal_len + act_candidate_len, device=device)
                            for _ in range(request.batch_size)
                        ],
                        dim=0,
                    )
                    for _ in range(request.mc_num // request.batch_size)
                ]

                seq = torch.concatenate([goal_tokens, prompt_tokens, act_candidate_tokens])[None, :]
                seq = seq.repeat((request.batch_size, 1)).to(device)
                prompt_index = torch.cat(
                    (
                        torch.zeros(goal_len, dtype=torch.bool, device=device),
                        torch.ones(prompt_len, dtype=torch.bool, device=device),
                        torch.zeros(act_candidate_len, dtype=torch.bool, device=device),
                    ),
                    dim=0,
                )
                score = get_joint_log_likelihood(
                    model,
                    seq,
                    prompt_index,
                    shuffled_seq,
                    k,
                    mc_num=request.mc_num,
                    batch_size=request.batch_size,
                    cfg_scale=LOG_LIKELIHOOD_CFG_SCALE,
                    mask_id=LOG_LIKELIHOOD_MASK_ID,
                )
                all_scores.append(float(score))

        normalized_scores = torch.softmax(
            torch.tensor(all_scores, dtype=torch.float64),
            dim=0,
        ).detach().cpu().tolist()
    
        logger.info(
            "Loglikelihood complete: candidates=%d mc_num=%d batch_size=%d latency=%.3fs",
            len(request.reflect_list),
            request.mc_num,
            request.batch_size,
            time.perf_counter() - start_time,
        )

        return {"normalized_loglikelihoods": normalized_scores}
    
    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Loglikelihood failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "traceback": tb},
        ) from exc

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve_agentboard_llada:app", host="0.0.0.0", port=8000)
