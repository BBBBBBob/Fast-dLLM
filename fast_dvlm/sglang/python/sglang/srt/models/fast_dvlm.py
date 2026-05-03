# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fast-dVLM model for SGLang.

This model combines the Fast-dLLM v2 diffusion language model with the
Qwen2.5-VL vision encoder to support multimodal (image + text) diffusion
generation. The language backbone uses ENCODER_ONLY attention for block
diffusion decoding, while the vision encoder is identical to Qwen2.5-VL.

Architecture name: Fast_dLLM_Qwen2_5_VLForConditionalGeneration
"""

import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
)

from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from sglang.srt.models.utils import RotaryPosMixin, WeightsMapper, permute_inv
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix


def _patch_attention_to_encoder_only(model: Qwen2Model):
    """Patch all attention layers in Qwen2Model to use ENCODER_ONLY attention.

    This is the key change for dLLM: bidirectional attention instead of causal.
    """
    for layer in model.layers:
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'attn'):
            layer.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  NVFP4 W4A4 activation fake-quant support
#
#  When activation_scales.pt exists alongside the model checkpoint,
#  we install forward pre-hooks on Linear layers that fake-quantize
#  activations to NVFP4 (E2M1) and dequantize back, matching the
#  SM120 real W4A4 path (sgl_kernel::scaled_fp4_quant + fp4_gemm).
# ------------------------------------------------------------------ #
_FP4_E2M1_MAX = 6.0
_FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
_NVFP4_BLOCK = 16


def _safe_reciprocal(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x == 0, torch.zeros_like(x), 1.0 / x)


def _cast_to_e2m1_levels(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    ax = x.abs()
    out = torch.zeros_like(ax)
    out = torch.where(ax > 5.0, torch.full_like(ax, 6.0), out)
    out = torch.where((ax >= 3.5) & (ax <= 5.0), torch.full_like(ax, 4.0), out)
    out = torch.where((ax > 2.5) & (ax < 3.5), torch.full_like(ax, 3.0), out)
    out = torch.where((ax >= 1.75) & (ax <= 2.5), torch.full_like(ax, 2.0), out)
    out = torch.where((ax > 1.25) & (ax < 1.75), torch.full_like(ax, 1.5), out)
    out = torch.where((ax >= 0.75) & (ax <= 1.25), torch.full_like(ax, 1.0), out)
    out = torch.where((ax > 0.25) & (ax < 0.75), torch.full_like(ax, 0.5), out)
    return sign * out


def _nvfp4_fake_quant_act(x_2d: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """NVFP4 fake quant for activations. Clamps per-block scale to FP8 range
    before casting to match CUDA hardware saturation behavior."""
    orig_dtype = x_2d.dtype
    m, n = x_2d.shape
    x_rs = x_2d.reshape(m, n // _NVFP4_BLOCK, _NVFP4_BLOCK)
    vec_max = torch.abs(x_rs).amax(dim=-1, keepdim=True).to(torch.float32)

    scale = global_scale * (vec_max * _safe_reciprocal(torch.tensor(_FP4_E2M1_MAX, device=x_2d.device)))
    scale = scale.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)

    inv_global = _safe_reciprocal(global_scale)
    output_scale = _safe_reciprocal(scale * inv_global)

    scaled_x = x_rs.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -_FP4_E2M1_MAX, _FP4_E2M1_MAX).reshape(m, n)
    x_q = _cast_to_e2m1_levels(clipped_x)

    dequant_factor = scale / global_scale
    x_deq = x_q.reshape(m, n // _NVFP4_BLOCK, _NVFP4_BLOCK) * dequant_factor
    return x_deq.reshape(m, n).to(orig_dtype)


def _make_act_fq_hook(input_global_scale: torch.Tensor):
    def hook(module, args):
        x = args[0] if isinstance(args, tuple) else args
        if x is None or x.dim() < 2:
            return args
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1])
        m, n = x_2d.shape
        need_pad = n % _NVFP4_BLOCK != 0
        if need_pad:
            pad_n = _NVFP4_BLOCK - n % _NVFP4_BLOCK
            x_2d = F.pad(x_2d, (0, pad_n))
        x_fq = _nvfp4_fake_quant_act(x_2d, input_global_scale)
        if need_pad:
            x_fq = x_fq[:, :n]
        x_fq = x_fq.reshape(orig_shape).to(orig_dtype)
        if isinstance(args, tuple):
            return (x_fq,) + args[1:]
        return x_fq
    return hook


def _install_activation_fake_quant(model: nn.Module, model_path: str):
    """Load activation_scales.pt and install fake-quant hooks on Linear layers.

    Handles naming mismatches between HF (activation_scales.pt) and sglang:
      HF: model.language_model.layers.X.self_attn.q_proj  →  sglang: model.layers.X.self_attn.qkv_proj
      HF: model.language_model.layers.X.mlp.gate_proj     →  sglang: model.layers.X.mlp.gate_up_proj
    For fused layers (qkv_proj, gate_up_proj), we take the max scale across
    constituent projections since they share the same input activation.
    """
    scales_path = os.path.join(model_path, "activation_scales.pt")
    if not os.path.exists(scales_path):
        return 0
    device = next(model.parameters()).device
    act_scales = torch.load(scales_path, map_location=device, weights_only=True)

    def _find_scale(sglang_name: str):
        """Try to find a matching scale for a sglang module name."""
        # Direct match
        if sglang_name in act_scales:
            return act_scales[sglang_name]
        # sglang model.layers.X → HF model.language_model.layers.X
        hf_name = sglang_name.replace("model.layers.", "model.language_model.layers.")
        if hf_name in act_scales:
            return act_scales[hf_name]
        # Fused qkv_proj: take max scale (= min global_scale) across q/k/v
        # because they share the same input activation
        if "qkv_proj" in hf_name:
            candidates = []
            for proj in ["q_proj", "k_proj", "v_proj"]:
                key = hf_name.replace("qkv_proj", proj)
                if key in act_scales:
                    candidates.append(act_scales[key])
            if candidates:
                # Smaller global_scale = larger amax = more conservative
                return min(candidates, key=lambda x: x.item())
        # Fused gate_up_proj: take max across gate/up
        if "gate_up_proj" in hf_name:
            candidates = []
            for proj in ["gate_proj", "up_proj"]:
                key = hf_name.replace("gate_up_proj", proj)
                if key in act_scales:
                    candidates.append(act_scales[key])
            if candidates:
                return min(candidates, key=lambda x: x.item())
        # lm_head
        if sglang_name == "lm_head" and "lm_head" in act_scales:
            return act_scales["lm_head"]
        return None

    count = 0
    for name, module in model.named_modules():
        if "visual" in name:
            continue
        scale = _find_scale(name)
        if scale is None:
            continue
        # Match any module with a weight parameter (nn.Linear,
        # ColumnParallelLinear, RowParallelLinear, etc.)
        if hasattr(module, "weight"):
            module.register_forward_pre_hook(_make_act_fq_hook(scale.to(device)))
            count += 1
    logger.info(f"W4A4 NVFP4 activation fake-quant: installed hooks on {count} layers")
    return count


# ------------------------------------------------------------------ #
#  Hadamard rotation support
#
#  When hadamard_rotation.json exists, install forward pre-hooks that
#  rotate activation: x_rot = x @ H^T before each quantized Linear.
#  The weight is already rotated offline (W_rot = W @ H^T in checkpoint).
# ------------------------------------------------------------------ #
def _hadamard_matrix(n: int, device="cpu") -> torch.Tensor:
    assert n > 0 and (n & (n - 1)) == 0
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n ** 0.5)


def _get_hadamard(n: int, device="cpu") -> torch.Tensor:
    if n > 0 and (n & (n - 1)) == 0:
        return _hadamard_matrix(n, device)
    block = 1
    while block * 2 <= n and n % (block * 2) == 0:
        block *= 2
    block = min(block, 256)
    while n % block != 0:
        block //= 2
    H_b = _hadamard_matrix(block, device)
    return torch.block_diag(*[H_b for _ in range(n // block)])


def _make_hadamard_hook(H_t: torch.Tensor):
    """Pre-hook: x_rot = x @ H^T."""
    def hook(module, args):
        x = args[0] if isinstance(args, tuple) else args
        if x is None or x.dim() < 2:
            return args
        x_rot = (x.float() @ H_t).to(x.dtype)
        return (x_rot,) + args[1:] if isinstance(args, tuple) else x_rot
    return hook


def _install_quarot_rotation(model: nn.Module, model_path: str):
    """Load quarot_config.json and set online Hadamard rotation attributes.

    QuaRot only needs 2 online Hadamard transforms per layer:
      - Before o_proj: partial Hadamard (head-wise, head_dim sized)
      - Before down_proj: full Hadamard (intermediate_size)
    All other rotations are absorbed into weights offline.
    """
    import json as _json
    config_path = os.path.join(model_path, "quarot_config.json")
    if not os.path.exists(config_path):
        return 0
    with open(config_path) as f:
        config = _json.load(f)
    if config.get("type") != "quarot":
        return 0

    device = next(model.parameters()).device
    head_dim = config["head_dim"]
    intermediate_size = config["intermediate_size"]
    num_heads = config["num_heads"]
    fp32_had = config.get("fp32_had", True)
    down_K = config.get("down_K", 1)
    o_K = config.get("o_K", 1)

    # Load hadK matrices if needed
    hadK_path = os.path.join(model_path, "quarot_hadK.pt")
    hadK_dict = {}
    if os.path.exists(hadK_path):
        hadK_dict = torch.load(hadK_path, map_location="cpu", weights_only=True)

    count = 0
    for name, module in model.named_modules():
        if "visual" in name:
            continue
        module_type = type(module).__name__

        if module_type == "Qwen2Attention":
            # o_proj needs partial Hadamard (head-wise)
            # o_proj: FULL Hadamard on input dim (hidden_size), NOT per-head
            module._quarot_o_dim = head_dim * num_heads  # = hidden_size
            module._quarot_fp32_had = fp32_had
            count += 1

        elif module_type == "Qwen2MLP":
            # down_proj needs full Hadamard
            module._quarot_down_dim = intermediate_size
            module._quarot_down_K = down_K
            _hk = hadK_dict.get("down_hadK", None)
            module._quarot_down_hadK = _hk.to(device) if _hk is not None else None
            module._quarot_fp32_had = fp32_had
            count += 1

    logger.info(f"QuaRot rotation: configured {count} modules for online Hadamard")
    return count


def _install_hadamard_rotation(model: nn.Module, model_path: str):
    """Load hadamard_rotation.json and set Hadamard rotation matrices on modules.

    Instead of Python hooks (which don't work with CUDA graph / torch.compile),
    we set _had_Ht_* attributes on Qwen2Attention and Qwen2MLP modules.
    The forward methods in qwen2.py check for these attributes inline.
    """
    import json as _json
    config_path = os.path.join(model_path, "hadamard_rotation.json")
    if not os.path.exists(config_path):
        return 0
    device = next(model.parameters()).device
    with open(config_path) as f:
        config = _json.load(f)
    rotated_layers = config.get("rotated_layers", {})
    if not rotated_layers:
        return 0

    count = 0
    for name, module in model.named_modules():
        if "visual" in name:
            continue
        module_type = type(module).__name__

        if module_type == "Qwen2Attention":
            hf_q = name.replace("model.layers.", "model.language_model.layers.") + ".q_proj"
            if hf_q in rotated_layers:
                module._had_dim_qkv = rotated_layers[hf_q]
                count += 1
            hf_o = name.replace("model.layers.", "model.language_model.layers.") + ".o_proj"
            if hf_o in rotated_layers:
                module._had_dim_o = rotated_layers[hf_o]
                count += 1

        elif module_type == "Qwen2MLP":
            hf_gate = name.replace("model.layers.", "model.language_model.layers.") + ".gate_proj"
            if hf_gate in rotated_layers:
                module._had_dim_gate_up = rotated_layers[hf_gate]
                count += 1
            hf_down = name.replace("model.layers.", "model.language_model.layers.") + ".down_proj"
            if hf_down in rotated_layers:
                module._had_dim_down = rotated_layers[hf_down]
                count += 1

    logger.info(f"Hadamard rotation (fast kernel): set dims on {count} module attributes")
    return count


class FastDVLMForConditionalGeneration(nn.Module):
    """
    Fast-dVLM: Fast Diffusion Vision-Language Model for SGLang.

    Combines:
    - Qwen2.5-VL vision encoder (Qwen2_5_VisionTransformer)
    - FastDLLMV2Model language backbone (ENCODER_ONLY attention for dLLM)
    - Full logits return for diffusion decoding
    """

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_up_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # Weight mapping: HF checkpoint names -> sglang names
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "attn.qkv": "attn.qkv_proj",
        },
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        },
    )

    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.pp_group = get_pp_group()
        self.config = config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        # Language model: Qwen2Model with ENCODER_ONLY attention for dLLM.
        # Uses Qwen2Model (not FastDLLMV2Model) because it supports MRoPE
        # which is required by Qwen2.5-VL's 3D positional encoding.
        if not self.config.encoder_only:
            self.model = Qwen2Model(
                config,
                quant_config=quant_config,
                prefix=add_prefix("model", prefix),
            )
            # NOTE: Do NOT patch attention to ENCODER_ONLY here.
            # The dLLM algorithm (HierarchyBlock) dynamically switches between:
            #   - DECODER (causal) for prefill and KV cache writes
            #   - ENCODER_ONLY (bidirectional) for block denoising iterations
            # Prefill (EXTEND mode) uses default DECODER attention, which is correct.

            # LM head
            if self.pp_group.is_last_rank:
                if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:
                    self.lm_head = self.model.embed_tokens
                else:
                    self.lm_head = ParallelLMHead(
                        self.config.vocab_size,
                        self.config.hidden_size,
                        quant_config=quant_config,
                        prefix=add_prefix("lm_head", prefix),
                    )
            else:
                self.lm_head = PPMissingLayer()

            # PP weight tying
            if self.pp_group.world_size > 1 and config.tie_word_embeddings:
                if self.pp_group.is_first_rank:
                    self.pp_group.send(
                        self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                    )
                elif self.pp_group.is_last_rank:
                    emb_token_weight = self.pp_group.recv(
                        size=(config.vocab_size, config.hidden_size),
                        dtype=next(self.model.parameters()).dtype,
                        src=self.pp_group.first_rank,
                    )
                    self.lm_head.weight.copy_(emb_token_weight)
        else:
            self.lm_head = None

        # Vision encoder: same as Qwen2.5-VL
        self.visual = Qwen2_5_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
            use_data_parallel=self.use_data_parallel,
            max_context_len=self.config.max_position_embeddings,
        )

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        # Return full logits for dLLM diffusion decoding
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)

        expected_dim = getattr(self.visual, "embed_dim", -1)
        if expected_dim == -1:
            vision_conf = self.config.vision_config
            expected_dim = getattr(
                vision_conf, "embed_dim", getattr(vision_conf, "hidden_size", -1)
            )

        raw_patch_dim = 1176

        if pixel_values.dim() == 2:
            current_dim = pixel_values.shape[-1]
            if current_dim == expected_dim:
                return pixel_values
            if current_dim != raw_patch_dim:
                return pixel_values

        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, image_grid_thw.tolist(), rope_type="rope_3d"
            )
        else:
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, video_grid_thw.tolist(), rope_type="rope_3d"
            )
        else:
            video_embeds = self.visual(pixel_values, grid_thw=video_grid_thw)
        return video_embeds

    def post_process(
        self,
        inputs_embeds,
        modalities: List[Modality],
        embeddings: List[torch.Tensor],
        indices: List[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        new_embeddings = []
        for i, (modality, embedding, index) in enumerate(
            zip(modalities, embeddings, indices)
        ):
            if embedding is None or index is None:
                continue
            new_embeddings.append(embedding)
        return new_embeddings, forward_batch

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embedding(input_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """Run forward pass for Fast-dVLM.

        Uses MRoPE positions (same as Qwen2.5-VL) for all modes.
        Prompt prefill (EXTEND): vision embeddings injected via general_mm_embed_routine.
        Block decode (DLLM_EXTEND): language model only, no multimodal processing.
        """
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_dllm_extend()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        if forward_batch.forward_mode.is_dllm_extend():
            # dLLM block decode: vision tokens already prefilled in KV cache.
            hidden_states = self.model(
                input_ids, positions, forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        else:
            # Prompt prefill or decode: process multimodal inputs
            hidden_states = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.model,
                multimodal_model=self,
                positions=positions,
                pp_proxy_tensors=pp_proxy_tensors,
            )

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    _lora_pattern = re.compile(
        r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)$"
    )

    def should_apply_lora(self, module_name: str) -> bool:
        return bool(self._lora_pattern.match(module_name))

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Vision encoder uses bf16 (matching checkpoint weights) for consistency
        # with the original HF implementation. All weights are stored in bf16.

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            name = name.replace("model.language_model.", "model.")
            if name.startswith("model.visual."):
                name = name[len("model."):]

            if self.pp_group.is_last_rank and "model.embed_tokens.weight" in name:
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    weight_loader(lm_head_param, loaded_weight)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if (
                    "visual" in name
                    and "up_proj" not in name
                    and "gate_proj" not in name
                ):
                    continue
                name = name.replace(weight_name, param_name)
                layer_id = get_layer_id(name)
                if (
                    layer_id is not None
                    and hasattr(self, "model")
                    and hasattr(self.model, "start_layer")
                    and (
                        layer_id < self.model.start_layer
                        or layer_id >= self.model.end_layer
                    )
                ):
                    continue

                if name.endswith(".bias") and name not in params_dict:
                    continue
                if (
                    self.config.encoder_only or self.config.language_only
                ) and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")

                try:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name in params_dict.keys():
                        param = params_dict[name]
                    else:
                        continue
                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        # Install runtime hooks if checkpoint contains them
        server_args = get_global_server_args()
        if server_args and hasattr(server_args, "model_path"):
            # W4A4 activation fake-quant hooks
            count = _install_activation_fake_quant(self, server_args.model_path)
            if count > 0:
                print(f"[FastDVLM] W4A4 activation fake-quant: {count} hooks installed")
            # QuaRot rotation
            count = _install_quarot_rotation(self, server_args.model_path)
            if count > 0:
                print(f"[FastDVLM] QuaRot rotation: {count} modules configured")
            # Legacy Hadamard rotation hooks
            count = _install_hadamard_rotation(self, server_args.model_path)
            if count > 0:
                print(f"[FastDVLM] Hadamard rotation: {count} hooks installed")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


# Register with HuggingFace architecture name from config.json
FastDVLMForConditionalGeneration.__name__ = "Fast_dVLMForConditionalGeneration"

EntryClass = FastDVLMForConditionalGeneration
