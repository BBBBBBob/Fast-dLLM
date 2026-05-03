from typing import Optional, Tuple, Union
import logging

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class HierarchyBlock(DllmAlgorithm):
    """Fast dLLM v2 hierarchical block decoding with token inheritance.

    Attention type switching (matches original HF implementation):
      - Prefill (EXTEND mode):          DECODER (causal)   — handled by model default
      - Denoising iterations:            ENCODER_ONLY (bidirectional) — can use CUDA Graph
      - Final forward (KV cache write):  DECODER (causal)   — always eager (no CUDA Graph)
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.9)
        self.sub_block_size = config.algorithm_config.get("sub_block_size", 8)
        self.token_shift = config.algorithm_config.get("token_shift", 1)
        self.debug = config.algorithm_config.get("debug", True)
        self.use_AR_for_first_token = config.algorithm_config.get("use_AR_for_first_token", True)

        self.last_inherited_token = None
        self.last_block_end_position = None
        self.tokenizer = None

    @staticmethod
    def _set_attention_type(model_runner: ModelRunner, attn_type: AttentionType):
        """Switch attention type on all language model layers."""
        model = model_runner.model
        # FastDVLMForConditionalGeneration.model is Qwen2Model
        layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        if layers is None:
            return
        for layer in layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'attn'):
                layer.self_attn.attn.attn_type = attn_type

    def _decode_tokens(self, token_ids, model_runner):
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                if hasattr(model_runner, 'model_config') and hasattr(model_runner.model_config, 'hf_config'):
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_runner.model_config.hf_config._name_or_path,
                        trust_remote_code=True
                    )
            except Exception as e:
                logger.warning(f"[HierarchyBlock] Failed to load tokenizer: {e}")
                self.tokenizer = False

        if self.tokenizer and self.tokenizer is not False:
            try:
                return self.tokenizer.decode(token_ids, skip_special_tokens=False)
            except Exception as e:
                return f"<decode_error: {e}>"
        return None

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        total_len = len(forward_batch.input_ids)
        block_mask = forward_batch.input_ids == self.mask_id
        num_masked = block_mask.sum().item()
        num_sub_blocks = total_len // self.sub_block_size
        block_start = total_len - num_masked

        # Detect new request (positions start from 0) and clear inheritance
        is_new_request = False
        if hasattr(forward_batch, 'positions') and forward_batch.positions is not None:
            if forward_batch.positions[0] == 0:
                is_new_request = True
                self.last_inherited_token = None
                self.last_block_end_position = None

        # Handle token inheritance for all-mask blocks
        first_token_is_mask = forward_batch.input_ids[0] == self.mask_id
        if first_token_is_mask and self.last_inherited_token is not None and not is_new_request:
            forward_batch.input_ids[0] = self.last_inherited_token

        # === Denoising iterations: use bidirectional attention ===
        # These can run with CUDA Graph since attention type stays constant.
        self._set_attention_type(model_runner, AttentionType.ENCODER_ONLY)

        # Process sub-blocks
        for sub_idx in range(num_sub_blocks):
            rel_start = sub_idx * self.sub_block_size
            rel_end = rel_start + self.sub_block_size

            while True:
                sub_mask = forward_batch.input_ids[rel_start:rel_end] == self.mask_id
                if sub_mask.sum().item() == 0:
                    break

                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
                logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
                full_logits = logits_output.full_logits
                assert full_logits is not None

                # Token shift: [L0, L0, L1, ..., LN-2]
                if self.token_shift > 0:
                    shifted_full = torch.cat([full_logits[:1], full_logits[:-1]], dim=0)
                else:
                    shifted_full = full_logits

                sub_logits = shifted_full[rel_start:rel_end, :]

                # Compute predictions and confidence
                preds = sub_logits.argmax(dim=-1)
                probs = F.softmax(sub_logits, dim=-1)
                conf = probs.gather(dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)
                conf = torch.where(sub_mask, conf, torch.tensor(-np.inf, device=conf.device))

                # Confidence-based unmask
                unmask = conf > self.threshold
                if unmask.sum().item() == 0:
                    unmask[conf.argmax()] = True
                unmask = unmask & sub_mask

                # Update tokens
                forward_batch.input_ids[rel_start:rel_end] = torch.where(
                    unmask, preds, forward_batch.input_ids[rel_start:rel_end]
                )

        # === Final forward: switch to causal attention for KV cache write ===
        # Must bypass CUDA Graph here because attention type changed.
        # Use forward_extend directly (eager mode) instead of forward()
        # which would route through the CUDA Graph runner.
        self._set_attention_type(model_runner, AttentionType.DECODER)

        logits_output = model_runner.forward_extend(forward_batch, pp_proxy_tensors=None)
        if isinstance(logits_output, tuple):
            logits_output = logits_output[0]
        full_logits = logits_output.full_logits

        if full_logits is not None:
            self.last_inherited_token = full_logits[-1].argmax().item()

        # Update position tracking
        if hasattr(forward_batch, 'positions') and forward_batch.positions is not None:
            self.last_block_end_position = forward_batch.positions[-1].item()

        # Restore ENCODER_ONLY for next block's denoising iterations
        # so CUDA Graph (captured with ENCODER_ONLY) stays valid.
        self._set_attention_type(model_runner, AttentionType.ENCODER_ONLY)

        return logits_output, forward_batch.input_ids[block_start:], can_run_cuda_graph


Algorithm = HierarchyBlock
