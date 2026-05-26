#!/usr/bin/env python
# coding=utf-8
"""Fast-dDrive finetuning entry (Section-Aware Structured Diffusion).

Mirrors ``fast_dvlm/train_scripts/finetune_dvlm.py`` and adds the SASD wiring:
deep JSON scaffold (frozen template tokens), Section-Importance-Weighted Loss
and Section-Adaptive Noise Schedule. Dataset is LMFlow's ``custom_multi_modal``
(LLaVA-style JSON on disk with ``image`` + ``conversations``).

Run via ``fast_ddrive/train_scripts/train_waymo_sasd.sh`` (a DeepSpeed
launcher that passes the paper-canonical SASD knobs). All CLI flags are
defined by LMFlow's ``ModelArguments`` / ``MultiModalDatasetArguments`` and
the standard ``TrainingArguments``.
"""

import json
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_third_party = os.path.join(_repo_root, "third_party")
if _third_party not in sys.path:
    sys.path.insert(0, _third_party)

from transformers import AutoProcessor, HfArgumentParser

from lmflow.args import AutoArguments, ModelArguments, MultiModalDatasetArguments
from lmflow.datasets.dataset import Dataset
# Fast-dDrive ships its own ``CustomMultiModalDataset`` + ``DataCollatorForQwenVL``
# variants alongside upstream's, so the deep JSON scaffold padding and bd_size /
# block_causal_no_dynamic collation are available without modifying upstream.
from lmflow.datasets.multi_modal_dataset_fast_ddrive import (
    CustomMultiModalDataset as _FastDDriveMultiModalDataset,
    DataCollatorForQwenVL,
)
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline


def _set_attr_both(model, name, value):
    """Set ``name`` on the backend model **and** its config so checkpoints round-trip."""
    setattr(model.backend_model, name, value)
    setattr(model.backend_model.config, name, value)


def _configure_mdm(model, model_args, data_args):
    """Apply MDM + deep-scaffold + SASD knobs onto the loaded backend model."""
    _set_attr_both(model, "bd_size", model_args.bd_size)
    language_model = getattr(getattr(model.backend_model, "model", None), "language_model", None)
    if language_model is not None:
        language_model.mask_id = model.tokenizer.encode("|<MASK>|")[0]

    # Register <|NULL|> token used to pad variable-length sections in the scaffold.
    null_ids = model.tokenizer.encode("<|NULL|>", add_special_tokens=False)
    if len(null_ids) != 1:
        model.tokenizer.add_tokens(["<|NULL|>"], special_tokens=True)
        null_id = model.tokenizer.encode("<|NULL|>", add_special_tokens=False)[0]
        print(f"<|NULL|> token registered: id={null_id}", flush=True)

    flag_defaults = {
        "complementary_mask": False,
        "always_mask_im_end": False,
        "flexible_bd_size": False,
        "use_block_causal_mask": False,
        "anneal_block_size": False,
        "minimum_noise_level": 1e-3,
        "entropy_loss": False,
        "entropy_loss_weight": 1.0,
        "enable_efficient_vision_embed": False,
        "block_causal_no_dynamic": False,
    }
    for attr, default in flag_defaults.items():
        _set_attr_both(model, attr, getattr(model_args, attr, default))

    if getattr(model_args, "cp_size", 1) > 1:
        raise NotImplementedError(
            "Context-parallel (cp_size > 1) is not shipped with this release; train with cp_size=1."
        )
    if getattr(model_args, "use_section_moe_lora", False):
        raise NotImplementedError(
            "Section-MoE-LoRA is not shipped with this release; use standard PEFT LoRA (--use_lora 1) instead."
        )

    # Deep JSON scaffold v2 — freezes structural JSON tokens during MDM denoising.
    deep_scaffold = bool(getattr(model_args, "deep_json_scaffold", False))
    _set_attr_both(model, "deep_json_scaffold", deep_scaffold)
    _set_attr_both(model, "use_json_scaffold", deep_scaffold)
    if deep_scaffold:
        print("Deep JSON scaffold v2 enabled: structural tokens frozen during denoising", flush=True)

    _set_attr_both(
        model, "section_token_budgets",
        json.loads(model_args.section_token_budgets) if getattr(model_args, "section_token_budgets", None) else None,
    )

    # SASD: per-section CE loss weights.
    weights_str = getattr(model_args, "section_loss_weights", None)
    weights = json.loads(weights_str) if weights_str else None
    _set_attr_both(model, "section_loss_weights", weights)
    if weights:
        print(f"SASD section loss weights enabled: {weights}", flush=True)

    # SASD: per-section Beta(α, β) noise schedule.
    schedule_str = getattr(model_args, "section_noise_schedule", None)
    schedule = json.loads(schedule_str) if schedule_str else None
    _set_attr_both(model, "section_noise_schedule", schedule)
    if schedule:
        print(f"SASD section noise schedule enabled: {schedule}", flush=True)

    data_args.bd_size = model_args.bd_size
    data_args._set_section_tokenizer = deep_scaffold


def main():
    parser = HfArgumentParser((
        ModelArguments,
        MultiModalDatasetArguments,
        AutoArguments.get_pipeline_args_class("finetuner"),
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, pipeline_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    data_args.return_as_qwen_messages = True

    finetuner = AutoPipeline.get_pipeline(
        pipeline_name="finetuner",
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )

    model = AutoModel.get_model(model_args)
    data_args.tokenizer = model.tokenizer

    dataset = Dataset(data_args, backend="custom_multi_modal")
    # Swap upstream's ``CustomMultiModalDataset`` for the Fast-dDrive variant
    # (deep JSON scaffold padding + ``register_tokenizer`` for explanation
    # NULL padding).  Upstream's wrapper around the LMFlow ``Dataset`` instance
    # is unchanged; only the backend dataset object is replaced.
    dataset.backend_dataset = _FastDDriveMultiModalDataset(data_args.dataset_path, data_args)

    if model_args.mdm:
        _configure_mdm(model, model_args, data_args)
    else:
        data_args.bd_size = None

    if getattr(model_args, "freeze_vision_encoder", False):
        for param in model.backend_model.model.visual.parameters():
            param.requires_grad = False
        print("Froze vision encoder", flush=True)

    processor_id = model_args.tokenizer_name or model_args.model_name_or_path
    processor = AutoProcessor.from_pretrained(
        processor_id,
        trust_remote_code=True,
        max_pixels=int(os.environ.get("MAX_PIXELS", 200704)),
        min_pixels=int(os.environ.get("MIN_PIXELS", 200704)),
    )
    processor.tokenizer = model.tokenizer  # keep a single tokenizer instance

    raw_ds = getattr(dataset, "backend_dataset", None)
    if raw_ds is not None and hasattr(raw_ds, "register_tokenizer"):
        raw_ds.register_tokenizer(model.tokenizer)

    if getattr(data_args, "_set_section_tokenizer", False):
        model.backend_model._section_tokenizer = model.tokenizer

    data_collator = DataCollatorForQwenVL(
        processor=processor,
        tokenizer=model.tokenizer,
        bd_size=model_args.bd_size if model_args.mdm else None,
        max_length=getattr(data_args, "max_length", None),
        block_causal_no_dynamic=getattr(model_args, "block_causal_no_dynamic", False),
    )

    finetuner.tune(model=model, dataset=dataset, data_collator=data_collator)


if __name__ == "__main__":
    main()
