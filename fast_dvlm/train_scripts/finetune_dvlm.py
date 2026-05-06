#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""Fast-dVLM finetuning entry (multimodal MDM).

Uses LMFlow ``Dataset`` with ``backend="custom_multi_modal"`` (LLaVA-style JSON on disk:
``image`` + ``conversations``). Training always enables ``return_as_qwen_messages`` so
each row is turned into Qwen ``messages`` + :class:`~lmflow.datasets.multi_modal_dataset.DataCollatorForQwenVL`
(``pixel_values``, ``image_grid_thw``). MDM ``bd_size`` / mask id are configured on the model only.

Example data: ``bash fast_dvlm/data/download_example_dataset.sh allava`` then
``source .../ALLaVA-4V/source_training_env.sh``. Use ``JSON_ONLY=1`` for JSON only;
``IMAGE_CHUNKS=0`` to download one image zip chunk (~9GB) for a smoke test; default
fetches all chunks ``images_0.zip`` … ``images_9.zip`` on the Hub.

Run with ``PYTHONPATH`` set to ``Fast-dLLM/third_party`` (see ``finetune_multimodal_example.sh``),
or after ``pip install -e v2/``.
"""

import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_third_party = os.path.join(_repo_root, "third_party")
if _third_party not in sys.path:
    sys.path.insert(0, _third_party)

from transformers import HfArgumentParser, AutoProcessor

from lmflow.args import ModelArguments, MultiModalDatasetArguments, AutoArguments
from lmflow.datasets.dataset import Dataset
from lmflow.datasets.multi_modal_dataset import DataCollatorForQwenVL
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline


def main():
    PipelineArguments = AutoArguments.get_pipeline_args_class("finetuner")

    parser = HfArgumentParser((ModelArguments, MultiModalDatasetArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, pipeline_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    # Qwen2.5-VL / Fast-dVLM MDM expects processor outputs (pixel_values + image_grid_thw).
    data_args.return_as_qwen_messages = True

    finetuner = AutoPipeline.get_pipeline(
        pipeline_name="finetuner",
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )

    dataset = Dataset(data_args, backend="custom_multi_modal")
    model = AutoModel.get_model(model_args)

    processor_id = (
        getattr(model_args, "tokenizer_name", None)
        or getattr(model_args, "model_name_or_path", None)
    )
    processor = AutoProcessor.from_pretrained(
        processor_id,
        trust_remote_code=model_args.trust_remote_code,
        max_pixels=705600,
    )

    data_collator = DataCollatorForQwenVL(
        processor=processor,
        tokenizer=model.tokenizer,
    )

    finetuner.tune(model=model, dataset=dataset, data_collator=data_collator)


if __name__ == "__main__":
    main()
