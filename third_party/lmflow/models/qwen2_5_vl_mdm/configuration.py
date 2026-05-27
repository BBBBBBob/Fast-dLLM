
# limitations under the License.
from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation


class Fast_dLLM_Qwen2_5_VLVisionConfig(PretrainedConfig):
    model_type = "fast_dllm_qwen2_5_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=[7, 15, 23, 31],
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range


class Fast_dLLM_Qwen2_5_VLTextConfig(PretrainedConfig):

    model_type = "fast_dllm_qwen2_5_vl_for_causal_lm"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=8192,
        intermediate_size=29568,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=80,
        layer_types=None,
        attention_dropout=0.0,
        rope_scaling=None,
        image_token_id=None,
        video_token_id=None,
        bd_size=8,
        self_spec_inference_mode=None,
        block_length=None,
        use_block_causal_mask=False,
        complementary_mask=True,
        minimum_noise_level=1e-3,
        entropy_loss=False,
        entropy_loss_weight=1.0,
        block_causal_no_dynamic=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.bd_size = bd_size
        self.layer_types = layer_types
        self.use_block_causal_mask = use_block_causal_mask
        self.complementary_mask = complementary_mask
        self.minimum_noise_level = minimum_noise_level
        self.entropy_loss = entropy_loss
        self.entropy_loss_weight = entropy_loss_weight
        self.block_causal_no_dynamic = block_causal_no_dynamic
        self.self_spec_inference_mode = self_spec_inference_mode
        self.block_length = block_length
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # and change type from 'mrope' to 'default' because `mrope` does default RoPE calculations
        # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
        # TODO: @raushan update config in the hub
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Fast_dLLM_Qwen2_5_VLConfig(PretrainedConfig):

    model_type = "fast_dllm_qwen2_5_vl"
    sub_configs = {"vision_config": Fast_dLLM_Qwen2_5_VLVisionConfig, "text_config": Fast_dLLM_Qwen2_5_VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        enable_efficient_vision_embed=False,
        always_mask_im_end=False,
        flexible_bd_size=False,
        anneal_block_size=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.enable_efficient_vision_embed = enable_efficient_vision_embed
        self.always_mask_im_end = always_mask_im_end
        self.flexible_bd_size = flexible_bd_size
        self.anneal_block_size = anneal_block_size

        super().__init__(**kwargs)

    # def to_dict(self):
    #     output = super().to_dict()
    #     output.pop("auto_map", None)
    #     return output


__all__ = ["Fast_dLLM_Qwen2_5_VLConfig", "Fast_dLLM_Qwen2_5_VLTextConfig"]