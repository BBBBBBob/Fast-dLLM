# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class StepVideoT2VSamplingParams(SamplingParams):
    # Video parameters
    height: int = 720
    width: int = 1280
    num_frames: int = 81

    # Denoising stage
    guidance_scale: float = 9.0
    num_inference_steps: int = 50

    # neg magic and pos magic
    # pos_magic: str = "Ultra HD, HDR video, ambient lighting, Dolby Atmos, steady framing, fluid motion, realistic detail, professional composition, hyperreal, natural, vivid, ultra-crisp."
    # neg_magic: str = "Dark footage, low resolution, bad hands, text overlays, missing or extra fingers, crops, low quality, grainy, logos, watermarks, usernames, blur."
