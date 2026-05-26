"""Minimal Fast-dDrive config shim for LMFlow training.

The full model code (``modeling.py``, ``generation_utils.py``,
``section_utils.py``) lives on the Hugging Face Hub under the Fast-dDrive
release; we only keep the configuration classes here so that LMFlow's
config-normalization path (in ``lmflow.models.hf_model_mixin``) can
import them without requiring the user to download a checkpoint first.
At inference time the model itself is loaded via
``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``.
"""

from .configuration import (
    Fast_dLLM_Qwen2_5_VLConfig,
    Fast_dLLM_Qwen2_5_VLTextConfig,
    Fast_dLLM_Qwen2_5_VLVisionConfig,
)

__all__ = [
    "Fast_dLLM_Qwen2_5_VLConfig",
    "Fast_dLLM_Qwen2_5_VLTextConfig",
    "Fast_dLLM_Qwen2_5_VLVisionConfig",
]
