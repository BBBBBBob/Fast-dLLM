import copy
import logging
import os
import random
import sys
import time
import traceback
import types
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
V1_DIR = ROOT / "v1"
sys.path.insert(0, str(V1_DIR))

from dream.model.generation_utils import DreamGenerationMixin  # noqa: E402
from dream.model.generation_utils_block import (  # noqa: E402
    DreamGenerationMixin as DreamBlockGenerationMixin,
)
from dream.model.modeling_dream import DreamModel  # noqa: E402
from dream.forward_loglikelihood import get_forward_log_likelihood  # noqa: E402
from dream.reverse_loglikelihood import get_reverse_log_likelihood  # noqa: E402


MODEL_API_NAME = "Dream"
MODEL_HF_NAME = "Dream-org/Dream-v0-Instruct-7B"
GENERATION_ALGORITHM = "confidence_threshold"
DREAM_DEFAULT_SYSTEM_PREFIX = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("serve_agentboard_dream")

app = FastAPI(title="AgentBoard Dream Server")

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


SERVER_SEED = int(os.getenv("SEED", "42"))
_set_global_seed(SERVER_SEED)
logger.info("Global seed set to: %d", SERVER_SEED)


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
            detail=(
                f'Unsupported model "{model}". '
                f'Only "{MODEL_API_NAME}" is supported.'
            ),
        )


def _validate_generate_request(request: GenerateRequest) -> None:
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    if request.gen_length <= 0:
        raise HTTPException(status_code=400, detail="gen_length must be positive")
    if request.steps <= 0:
        raise HTTPException(status_code=400, detail="steps must be positive")
    if request.temperature < 0:
        raise HTTPException(status_code=400, detail="temperature must be non-negative")
    if request.block_size <= 0:
        raise HTTPException(status_code=400, detail="block_size must be positive")
    if request.threshold is not None and not 0.0 <= request.threshold <= 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    if request.dual_cache:
        if request.gen_length % request.block_size != 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "gen_length must be divisible by block_size when dual_cache "
                    "is enabled"
                ),
            )
        num_blocks = request.gen_length // request.block_size
        if request.steps % num_blocks != 0:
            raise HTTPException(
                status_code=400,
                detail="steps must be divisible by the number of generation blocks",
            )
    elif request.threshold is not None and request.gen_length % request.steps != 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "gen_length must be divisible by steps when confidence-threshold "
                "generation is enabled"
            ),
        )


def _validate_likelihood_request(request: LikelihoodRequest) -> None:
    if not request.goal:
        raise HTTPException(status_code=400, detail="goal must be provided")
    if not request.reflect_list:
        raise HTTPException(
            status_code=400,
            detail="reflect_list must contain at least one candidate",
        )
    if request.mc_num <= 0 or request.batch_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="mc_num and batch_size must be positive",
        )
    if request.mc_num % request.batch_size != 0:
        raise HTTPException(
            status_code=400,
            detail="mc_num must be divisible by batch_size",
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
                _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                    DreamModel.from_pretrained(
                        MODEL_HF_NAME,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                    )
                    .to(_device)
                    .eval()
                )
                logger.info("Model loaded in %.2fs", time.perf_counter() - start_time)
    return _model, _tokenizer, _device


def _encode_messages(
    tokenizer,
    messages: List[Message],
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        _messages_to_dicts(messages),
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _bind_generation_mixin(model, use_dual_cache: bool) -> None:
    generation_mixin = (
        DreamBlockGenerationMixin if use_dual_cache else DreamGenerationMixin
    )
    model.diffusion_generate = types.MethodType(
        generation_mixin.diffusion_generate,
        model,
    )
    model._sample = types.MethodType(generation_mixin._sample, model)


def _truncate_at_eos(generated_ids: torch.Tensor, eos_token_id) -> torch.Tensor:
    if eos_token_id is None:
        return generated_ids
    eos_token_ids = (
        {int(token_id) for token_id in eos_token_id}
        if isinstance(eos_token_id, (list, tuple, set))
        else {int(eos_token_id)}
    )
    for index, token_id in enumerate(generated_ids.tolist()):
        if token_id in eos_token_ids:
            return generated_ids[:index]
    return generated_ids


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
    input_ids, _ = _encode_messages(tokenizer, request.messages)
    num_tokens = int(input_ids.shape[1])
    logger.info("Token count: model=%s tokens=%d", request.model, num_tokens)
    return {"num_of_tokens": num_tokens}


def _generate(request: GenerateRequest) -> dict:
    try:
        model, tokenizer, device = _get_model_and_tokenizer()
        prompt, attention_mask = _encode_messages(tokenizer, request.messages)
        prompt = prompt.to(device)
        attention_mask = attention_mask.to(device)

        generation_algorithm = (
            GENERATION_ALGORITHM if request.threshold is not None else "entropy"
        )
        generation_kwargs = {
            "attention_mask": attention_mask,
            "max_new_tokens": request.gen_length,
            "output_history": False,
            "return_dict_in_generate": True,
            "steps": request.steps,
            "temperature": request.temperature,
            "top_p": None,
            "top_k": None,
            "alg": generation_algorithm,
            "alg_temp": 0.0,
            "threshold": request.threshold,
        }
        if request.dual_cache:
            generation_kwargs.update(
                block_length=request.block_size,
                dual_cache=True,
            )

        start_time = time.perf_counter()
        with torch.inference_mode():
            _bind_generation_mixin(model, request.dual_cache)
            output = model.diffusion_generate(prompt, **generation_kwargs)

        output_ids = output.sequences if hasattr(output, "sequences") else output
        generated_ids = output_ids[0, prompt.shape[1] :].detach().cpu()
        generated_ids = _truncate_at_eos(generated_ids, tokenizer.eos_token_id)
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        generated_tokens = _generated_token_count(generated_ids, tokenizer)
        latency = time.perf_counter() - start_time
        tokens_per_second = generated_tokens / latency if latency > 0 else 0.0

        logger.info(
            "Generation complete: model=%s prompt_tokens=%d generated_tokens=%d "
            "latency=%.3fs tokens_per_sec=%.2f dual_cache=%s block_size=%d "
            "steps=%d gen_length=%d threshold=%s alg=%s",
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
            generation_algorithm,
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


@app.post("/generate")
def generate_response(
    request: GenerateRequest,
    authorization: Optional[str] = Header(default=None),
):
    _validate_model(request.model)
    _validate_generate_request(request)
    del authorization
    return _generate(request)


def _tokenize_text(tokenizer, text: str) -> torch.Tensor:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    return input_ids[0]


def _render_chat(
    tokenizer,
    reflect_candidate: List[dict],
    add_generation_prompt: bool,
) -> str:
    return tokenizer.apply_chat_template(
        reflect_candidate,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )


def _require_special_token_id(tokenizer, token_name: str) -> int:
    token_id = getattr(tokenizer, f"{token_name}_token_id", None)
    if token_id is None:
        raise RuntimeError(f"Dream tokenizer does not define a {token_name} token")
    return int(token_id)


def _add_sequence_boundaries(
    tokenizer,
    prefix: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _add_bos_boundary(tokenizer, prefix), _add_eos_boundary(tokenizer, target)


def _add_bos_boundary(tokenizer, tokens: torch.Tensor) -> torch.Tensor:
    bos_token_id = _require_special_token_id(tokenizer, "bos")
    if tokens.numel() == 0 or tokens[0].item() != bos_token_id:
        tokens = torch.cat(
            [torch.tensor([bos_token_id], dtype=torch.long), tokens]
        )
    return tokens


def _add_eos_boundary(tokenizer, tokens: torch.Tensor) -> torch.Tensor:
    eos_token_id = _require_special_token_id(tokenizer, "eos")
    if tokens.numel() == 0 or tokens[-1].item() != eos_token_id:
        tokens = torch.cat(
            [tokens, torch.tensor([eos_token_id], dtype=torch.long)]
        )
    return tokens


def _prepare_reverse_pair(
    tokenizer,
    reflect_candidate: List[dict],
    goal: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    goal_tokens = _tokenize_text(tokenizer, goal)
    if goal_tokens.numel() == 0:
        raise HTTPException(
            status_code=400,
            detail="goal cannot tokenize to an empty sequence",
        )

    reflection = _render_chat(
        tokenizer,
        reflect_candidate,
        add_generation_prompt=False,
    )
    reflection_tokens = _tokenize_text(tokenizer, reflection)
    return _add_sequence_boundaries(tokenizer, goal_tokens, reflection_tokens)


def _prepare_forward_prefix(
    tokenizer,
    reflect_candidate: List[dict],
    goal: str,
) -> torch.Tensor:
    forward_reflect = copy.deepcopy(reflect_candidate)
    forward_reflect[0]["content"] = (
        f"{goal}\n{forward_reflect[0]['content']}"
    )
    prefix_text = _render_chat(
        tokenizer,
        forward_reflect[:-1],
        add_generation_prompt=False,
    )
    prefix_tokens = _tokenize_text(tokenizer, prefix_text)
    if prefix_tokens.numel() == 0:
        raise HTTPException(
            status_code=400,
            detail="goal cannot tokenize to an empty sequence",
        )
    return _add_bos_boundary(tokenizer, prefix_tokens)


def _prepare_forward_target(
    tokenizer,
    reflect_candidate: List[dict],
) -> torch.Tensor:
    target_text = _render_chat(
        tokenizer,
        reflect_candidate[-1:],
        add_generation_prompt=False,
    )
    if not target_text.startswith(DREAM_DEFAULT_SYSTEM_PREFIX):
        raise HTTPException(
            status_code=400,
            detail="Dream chat template did not include the expected system prefix",
        )
    target_text = target_text[len(DREAM_DEFAULT_SYSTEM_PREFIX) :]

    target_tokens = _tokenize_text(tokenizer, target_text)
    if target_tokens.numel() == 0:
        raise HTTPException(
            status_code=400,
            detail="assistant candidate cannot tokenize to an empty sequence",
        )
    return _add_eos_boundary(tokenizer, target_tokens)


def _make_mc_noise(
    request: LikelihoodRequest,
    sequence_length: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    num_mc_batches = request.mc_num // request.batch_size
    u0 = [
        torch.rand(1, device=device, dtype=torch.float32)
        for _ in range(num_mc_batches)
    ]
    return u0, _make_mc_mask_rand(request, sequence_length, device)


def _make_mc_mask_rand(
    request: LikelihoodRequest,
    sequence_length: int,
    device: torch.device,
) -> List[torch.Tensor]:
    num_mc_batches = request.mc_num // request.batch_size
    mask_rand = [
        torch.rand(
            (request.batch_size, sequence_length),
            device=device,
            dtype=torch.float32,
        )
        for _ in range(num_mc_batches)
    ]
    mask_rand = torch.concat(mask_rand, dim=0)
    return mask_rand
    #  return [
    #     torch.rand(
    #         (request.batch_size, sequence_length),
    #         device=device,
    #         dtype=torch.float32,
    #     )
    #     for _ in range(num_mc_batches)
    # ]


def _score_reverse_candidate(
    model,
    tokenizer,
    device: torch.device,
    request: LikelihoodRequest,
    reflect_candidate: List[dict],
) -> float:
    prefix, target = _prepare_reverse_pair(
        tokenizer,
        reflect_candidate,
        request.goal,
    )
    u0, mask_rand = _make_mc_noise(
        request,
        int(prefix.numel() + target.numel()),
        device,
    )
    return float(
        get_reverse_log_likelihood(
            model,
            prefix,
            target,
            request.mc_num,
            request.batch_size,
            u0,
            mask_rand,
            _require_special_token_id(tokenizer, "mask"),
            device,
        )
    )


def _score_forward_candidate(
    model,
    tokenizer,
    device: torch.device,
    request: LikelihoodRequest,
    prefix: torch.Tensor,
    target: torch.Tensor,
    u0: List[torch.Tensor],
    mask_rand: List[torch.Tensor],
) -> float:
    return float(
        get_forward_log_likelihood(
            model,
            prefix,
            target,
            request.mc_num,
            request.batch_size,
            u0,
            mask_rand,
            _require_special_token_id(tokenizer, "mask"),
            device,
        )
    )


def _run_likelihood(
    request: LikelihoodRequest,
    mode: str,
) -> dict:
    model, tokenizer, device = _get_model_and_tokenizer()
    reverse_scores = []
    forward_scores = []
    start_time = time.perf_counter()

    if mode in ("forward", "multi"):
        forward_prefix = _prepare_forward_prefix(
            tokenizer,
            request.reflect_list[0],
            request.goal,
        )
        forward_u0, forward_prefix_mask_rand = _make_mc_noise(
            request,
            int(forward_prefix.numel()),
            device,
        )
    if mode in ("reverse", "multi"):

    with torch.inference_mode():
        for reflect_candidate in request.reflect_list:
            if mode in ("reverse", "multi"):
                reverse_scores.append(
                    _score_reverse_candidate(
                        model,
                        tokenizer,
                        device,
                        request,
                        reflect_candidate,
                    )
                )
            if mode in ("forward", "multi"):
                target = _prepare_forward_target(tokenizer, reflect_candidate)
                target_length = int(target.numel())
                target_mask_rand = _make_mc_mask_rand(
                    request,
                    target_length,
                    device,
                )
                # candidate_mask_rand = [
                #     torch.cat(
                #         [prefix_noise, target_noise],
                #         dim=1,
                #     )
                #     for prefix_noise, target_noise in zip(
                #         forward_prefix_mask_rand,
                #         target_mask_rand,
                #     )
                # ]
                candidate_mask_rand = torch.concat([prefix_noise, target_noise], dim=1)
                forward_scores.append(
                    _score_forward_candidate(
                        model,
                        tokenizer,
                        device,
                        request,
                        forward_prefix,
                        target,
                        forward_u0,
                        candidate_mask_rand,
                    )
                )
    if mode == "reverse":
        score_tensor = torch.tensor(reverse_scores, dtype=torch.float64)
    elif mode == "forward":
        score_tensor = torch.tensor(forward_scores, dtype=torch.float64)
    elif mode == "multi":
        # This is equivalent to multiplying the separately normalized forward
        # and reverse scores and normalizing the product, without underflow.
        score_tensor = torch.tensor(reverse_scores, dtype=torch.float64)
        score_tensor += torch.tensor(forward_scores, dtype=torch.float64)
    else:
        raise ValueError(f"Unsupported likelihood mode: {mode}")

    if not torch.isfinite(score_tensor).all().item():
        raise RuntimeError(f"Dream {mode} likelihood produced a non-finite score")
    normalized_scores = torch.softmax(score_tensor, dim=0).tolist()

    logger.info(
        "%s loglikelihood complete: candidates=%d mc_num=%d batch_size=%d "
        "latency=%.3fs",
        mode.capitalize(),
        len(request.reflect_list),
        request.mc_num,
        request.batch_size,
        time.perf_counter() - start_time,
    )
    return {"normalized_loglikelihoods": normalized_scores}


def _likelihood_response(
    request: LikelihoodRequest,
    authorization: Optional[str],
    mode: str,
) -> dict:
    _validate_model(request.model)
    _validate_likelihood_request(request)
    del authorization

    try:
        return _run_likelihood(request, mode)
    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("%s loglikelihood failed: %s", mode.capitalize(), exc)
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "traceback": tb},
        ) from exc


@app.post("/reverse_loglikelihood")
def reverse_loglikelihood_response(
    request: LikelihoodRequest,
    authorization: Optional[str] = Header(default=None),
):
    return _likelihood_response(request, authorization, "reverse")


@app.post("/forward_loglikelihood")
def forward_loglikelihood_response(
    request: LikelihoodRequest,
    authorization: Optional[str] = Header(default=None),
):
    return _likelihood_response(request, authorization, "forward")


@app.post("/multi_loglikelihood")
def multi_loglikelihood_response(
    request: LikelihoodRequest,
    authorization: Optional[str] = Header(default=None),
):
    return _likelihood_response(request, authorization, "multi")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve_agentboard_dream:app", host="0.0.0.0", port=8000)
