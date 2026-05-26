#!/usr/bin/env python
# coding=utf-8
# FIXME update the doc string.
"""This Python code defines a class Multi Modal Dataset.
"""
import copy
from dataclasses import dataclass, field
import json
from PIL import Image
from typing import Optional
import os
import os.path as osp
import transformers
import time
import torch
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

from lmflow.args import DatasetArguments
from lmflow.utils import llava_conversation_lib as conversation_lib

from lmflow.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


@dataclass
class DataCollatorForQwenVL(object):
    """Collate function that performs Qwen2.5-VL preprocessing at batch time.

    - Applies chat template per sample
    - Extracts image/video inputs from messages via process_vision_info
    - Uses processor to perform resize/pad/normalize and tokenize
    - Builds labels and masks tokens before a target sequence
    - Optionally pads to a multiple of bd_size for downstream reshape needs
    """

    processor: transformers.PreTrainedTokenizer
    tokenizer: transformers.PreTrainedTokenizer
    bd_size: Optional[int] = None
    block_causal_no_dynamic: bool = False
    max_length: Optional[int] = None
    mask_token: str = "|<MASK>|"
    ignore_index: int = IGNORE_INDEX
    target_sequence: torch.Tensor = field(default_factory=lambda: torch.tensor([151644, 77091, 198], dtype=torch.long))

    def __call__(self, instances):
        # Optional lightweight profiling (disabled by default)
        # Enable with: DVLM_PROFILE=1, DVLM_PROFILE_EVERY=50, DVLM_PROFILE_MAX=20
        prof_enabled = os.environ.get("DVLM_PROFILE", "0").strip().lower() in {"1", "true", "yes", "on"}
        if prof_enabled:
            if not hasattr(self, "_profile_count"):
                self._profile_count = 0
                self._profile_seen = 0
            self._profile_count += 1
            prof_every = int(os.environ.get("DVLM_PROFILE_EVERY", "50"))
            prof_max = int(os.environ.get("DVLM_PROFILE_MAX", "20"))
            prof_do = (self._profile_seen < prof_max) and (self._profile_count % max(prof_every, 1) == 0)
        else:
            prof_do = False

        if prof_do:
            t_batch_start = time.perf_counter()
            t_build_start = t_batch_start

        # Collect chat texts and vision inputs
        texts = []
        batch_image_inputs = []
        batch_video_inputs = []
        
        # Filter out invalid instances
        valid_instances = []
        for ins in instances:
            messages = ins.get("messages", [])
            if not messages:
                print(f"Warning: Skipping instance with empty messages", flush=True)
                continue
            valid_instances.append(ins)
        
        # If all instances are invalid, create a minimal dummy batch
        if not valid_instances:
            print(f"Warning: All instances in batch are invalid, using dummy data", flush=True)
            valid_instances = [{
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello."}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]}
                ]
            }]
        
        for ins in valid_instances:
            messages = ins["messages"]
            try:
                text = self.processor.apply_chat_template(messages, tokenize=False)
                texts.append(text)
                image_inputs, video_inputs = process_vision_info(messages)
                batch_image_inputs.append(image_inputs)
                batch_video_inputs.append(video_inputs)
            except Exception as e:
                print(f"Warning: Failed to process instance: {e}, skipping", flush=True)
                # Add a dummy entry to keep batch size consistent
                texts.append("<|im_start|>user\nHello.<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n")
                batch_image_inputs.append(None)
                batch_video_inputs.append(None)

        if prof_do:
            t_build_end = time.perf_counter()
            t_build = t_build_end - t_build_start

        # Process each instance separately (or use batched processor fast path)
        all_input_ids = []
        all_attention_masks = []
        all_pixel_values = []
        all_image_grid_thw = []
        is_dummy = []  # Track which samples are dummy (exceeded max_length)
        
        # Get im_end token for padding
        im_end_token = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(im_end_token) == 1:
            im_end_id = im_end_token[0]
        else:
            # Fallback: use common Qwen im_end token ID
            im_end_id = 151645
        
        # Get mask token for bd_size padding
        mask_id = self.tokenizer.convert_tokens_to_ids(self.mask_token)
        
        # Prepare dummy input for samples that exceed max_length
        dummy_text = "<|im_start|>user\nHello.<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>\n"
        dummy_inputs = self.processor(text=[dummy_text], padding=False, return_tensors="pt")
        dummy_input_ids = dummy_inputs["input_ids"].squeeze(0)
        dummy_attention_mask = dummy_inputs.get("attention_mask", dummy_inputs["input_ids"].ne(self.tokenizer.pad_token_id)).squeeze(0)
        
        use_batch_processor = self.max_length is None
        if use_batch_processor:
            # Fast path: process the whole batch in one processor call
            try:
                proc_kwargs = {
                    "text": texts,
                    "padding": True,
                    "return_tensors": "pt",
                }
                has_images = any((imgs is not None and len(imgs) > 0) for imgs in batch_image_inputs)
                has_videos = any((vids is not None and len(vids) > 0) for vids in batch_video_inputs)
                if has_images:
                    proc_kwargs["images"] = [imgs if imgs is not None else [] for imgs in batch_image_inputs]
                if has_videos:
                    proc_kwargs["videos"] = [vids if vids is not None else [] for vids in batch_video_inputs]

                if prof_do:
                    t_proc_start = time.perf_counter()
                inputs = self.processor(**proc_kwargs)
                if prof_do:
                    t_proc = time.perf_counter() - t_proc_start
                    t_proc_sum = t_proc

                input_ids = inputs["input_ids"]
                attention_mask = inputs.get(
                    "attention_mask",
                    input_ids.ne(self.tokenizer.pad_token_id),
                )

                # Replace padding tokens with mask_id (or im_end_id) to align with MDM padding behavior
                if mask_id is not None:
                    input_ids = input_ids.clone()
                    input_ids[attention_mask == 0] = mask_id
                else:
                    input_ids = input_ids.clone()
                    input_ids[attention_mask == 0] = im_end_id

                # Align to bd_size so MDM reshape (L // bd_size, bd_size) works
                max_seq_len = input_ids.size(1)
                if isinstance(self.bd_size, int) and self.bd_size > 0:
                    remainder = max_seq_len % self.bd_size
                    if remainder != 0:
                        pad_len = self.bd_size - remainder
                        pad_ids = torch.full(
                            (input_ids.size(0), pad_len),
                            mask_id if mask_id is not None else im_end_id,
                            dtype=input_ids.dtype,
                            device=input_ids.device,
                        )
                        pad_attn = torch.zeros(
                            (attention_mask.size(0), pad_len),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )
                        input_ids = torch.cat([input_ids, pad_ids], dim=1)
                        attention_mask = torch.cat([attention_mask, pad_attn], dim=1)
                        max_seq_len = input_ids.size(1)

                labels = input_ids.clone()

                # Mask tokens for multi-turn conversations
                seq_pat = self.target_sequence
                T = int(seq_pat.numel())
                for b in range(input_ids.size(0)):
                    seq = input_ids[b]
                    labels[b, :] = self.ignore_index
                    i = 0
                    while i <= seq.numel() - T:
                        if torch.equal(seq[i:i + T], seq_pat.to(seq.device)):
                            response_start = i + T
                            response_end = response_start
                            while response_end < seq.numel():
                                if seq[response_end].item() == im_end_id:
                                    response_end += 2
                                    break
                                if response_end + T <= seq.numel() and torch.equal(seq[response_end:response_end + T], seq_pat.to(seq.device)):
                                    break
                                response_end += 1
                            if response_start < response_end:
                                labels[b, response_start:response_end] = seq[response_start:response_end]
                            i = response_end + 1
                        else:
                            i += 1

                # Mask padding positions in labels
                for b in range(input_ids.size(0)):
                    is_mask_pad = (input_ids[b] == mask_id)
                    labels[b, is_mask_pad] = self.ignore_index

                batch = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }

                if "pixel_values" in inputs:
                    batch["pixel_values"] = inputs["pixel_values"]
                if "image_grid_thw" in inputs:
                    batch["image_grid_thw"] = inputs["image_grid_thw"]
                else:
                    batch["image_grid_thw"] = torch.tensor([[0, 0, 0]], dtype=torch.long)

                if prof_do:
                    t_batch = time.perf_counter() - t_batch_start
                    num_images = 0
                    for imgs in batch_image_inputs:
                        if imgs is None:
                            continue
                        try:
                            num_images += len(imgs)
                        except Exception:
                            num_images += 1
                    print(
                        f"[DVLM_PROFILE] batch={input_ids.size(0)} images={num_images} max_seq={max_seq_len} "
                        f"build={t_build:.4f}s proc={t_proc:.4f}s (inner={t_proc_sum:.4f}s) "
                        f"labels=0.0000s concat=0.0000s total={t_batch:.4f}s",
                        flush=True,
                    )
                    self._profile_seen += 1

                return batch
            except Exception as e:
                print(f"Warning: Batched processor failed: {e}. Falling back to per-sample processing.", flush=True)

        if prof_do:
            t_proc_start = time.perf_counter()
            t_proc_sum = 0.0

        for idx, (text, image_inputs, video_inputs) in enumerate(zip(texts, batch_image_inputs, batch_video_inputs)):
            # Build processor kwargs for single instance
            proc_kwargs = {
                "text": [text],
                "padding": False,  # Don't pad yet, we'll do it manually later
                "return_tensors": "pt",
            }
            
            # Add images if present
            if image_inputs is not None and len(image_inputs) > 0:
                proc_kwargs["images"] = [image_inputs]
            
            # Add videos if present
            if video_inputs is not None and len(video_inputs) > 0:
                proc_kwargs["videos"] = [video_inputs]
            
            # Process single instance
            if prof_do:
                _t0 = time.perf_counter()
                inputs = self.processor(**proc_kwargs)
                t_proc_sum += (time.perf_counter() - _t0)
            else:
                inputs = self.processor(**proc_kwargs)
            
            # Extract outputs
            input_ids_single = inputs["input_ids"].squeeze(0)  # [seq_len]
            attention_mask_single = inputs.get("attention_mask", inputs["input_ids"].ne(self.tokenizer.pad_token_id)).squeeze(0)
            
            # Check if sample exceeds max_length (before bd_size padding)
            if self.max_length is not None and input_ids_single.size(0) > self.max_length:
                if 151655 in input_ids_single[self.max_length:]:
                    print(f"Warning: Sample {idx} exceeds max_length ({input_ids_single.size(0)} > {self.max_length}), using dummy", flush=True)
                    # Use dummy input instead
                    input_ids_single = dummy_input_ids.clone()
                    attention_mask_single = dummy_attention_mask.clone()
                    is_dummy.append(True)
                    # No pixel values for dummy
                    all_pixel_values.append(None)
                    all_image_grid_thw.append(torch.tensor([[0, 0, 0]], dtype=torch.long))
                else:
                    print(f"Warning: Sample {idx} exceeds max_length ({input_ids_single.size(0)} > {self.max_length}), using max_length", flush=True)
                    input_ids_single = input_ids_single[:self.max_length]
                    attention_mask_single = attention_mask_single[:self.max_length]
                    is_dummy.append(False)
                    if "pixel_values" in inputs:
                        # pixel_values shape: [num_images, channels, height, width] or [l, hidden_dim]
                        all_pixel_values.append(inputs["pixel_values"])
                    else:
                        # Pure text, no pixel values
                        all_pixel_values.append(None)
                    
                    # Handle image_grid_thw
                    if "image_grid_thw" in inputs:
                        all_image_grid_thw.append(inputs["image_grid_thw"])
                    else:
                        # Pure text, use [0, 0, 0]
                        all_image_grid_thw.append(torch.tensor([[0, 0, 0]], dtype=torch.long))
            else:
                is_dummy.append(False)
                # Handle pixel_values
                if "pixel_values" in inputs:
                    # pixel_values shape: [num_images, channels, height, width] or [l, hidden_dim]
                    all_pixel_values.append(inputs["pixel_values"])
                else:
                    # Pure text, no pixel values
                    all_pixel_values.append(None)
                
                # Handle image_grid_thw
                if "image_grid_thw" in inputs:
                    all_image_grid_thw.append(inputs["image_grid_thw"])
                else:
                    # Pure text, use [0, 0, 0]
                    all_image_grid_thw.append(torch.tensor([[0, 0, 0]], dtype=torch.long))
            
            # Optional pad to multiple of bd_size before adding to list
            if isinstance(self.bd_size, int) and self.bd_size > 0 and self.block_causal_no_dynamic:
                seq_len = input_ids_single.size(0)
                pad_len = (self.bd_size - (seq_len % self.bd_size)) % self.bd_size
                if pad_len > 0:
                    # Pad with mask_id for input_ids
                    pad_ids = torch.full((pad_len,), mask_id, dtype=input_ids_single.dtype, device=input_ids_single.device)
                    input_ids_single = torch.cat([input_ids_single, pad_ids], dim=0)
                    
                    # Pad with 1 for attention_mask (attend to padding)
                    pad_attn = torch.ones((pad_len,), dtype=attention_mask_single.dtype, device=attention_mask_single.device)
                    attention_mask_single = torch.cat([attention_mask_single, pad_attn], dim=0)
            
            all_input_ids.append(input_ids_single)
            all_attention_masks.append(attention_mask_single)

        if prof_do:
            t_proc_end = time.perf_counter()
            t_proc = t_proc_end - t_proc_start

        # Find max sequence length for padding
        max_seq_len = max(ids.size(0) for ids in all_input_ids)
        # When max_length is set, pad to exactly max_length (not just the
        # batch max).  This is critical for Context Parallel training where
        # all ranks in a CP group must have identical sequence lengths so
        # that NCCL all-gather operations don't hang on shape mismatches.
        if self.max_length is not None:
            max_seq_len = max(max_seq_len, self.max_length)
        # Align to bd_size so MDM reshape (L // bd_size, bd_size) works
        if isinstance(self.bd_size, int) and self.bd_size > 0:
            remainder = max_seq_len % self.bd_size
            if remainder != 0:
                max_seq_len += self.bd_size - remainder
        batch_size = len(all_input_ids)

        # Initialize padded tensors
        padded_input_ids = torch.full((batch_size, max_seq_len), mask_id if mask_id is not None else im_end_id, dtype=all_input_ids[0].dtype)
        padded_attention_masks = torch.zeros((batch_size, max_seq_len), dtype=all_attention_masks[0].dtype)

        # Fill in the actual values
        for b in range(batch_size):
            seq_len = all_input_ids[b].size(0)
            padded_input_ids[b, :seq_len] = all_input_ids[b]
            padded_attention_masks[b, :seq_len] = all_attention_masks[b]

        input_ids = padded_input_ids
        attention_masks = padded_attention_masks
        if prof_do:
            t_label_start = time.perf_counter()

        labels = input_ids.clone()

        # Mask tokens for multi-turn conversations
        # Only keep assistant responses, mask everything else (system, user inputs, format tokens)
        seq_pat = self.target_sequence  # <|im_start|>assistant\n
        T = int(seq_pat.numel())
        
        for b in range(input_ids.size(0)):
            seq = input_ids[b]
            # Start by masking everything
            labels[b, :] = self.ignore_index
            
            # Find all assistant response regions and unmask them
            i = 0
            while i <= seq.numel() - T:
                # Check if current position is <|im_start|>assistant\n
                if torch.equal(seq[i:i+T], seq_pat.to(seq.device)):
                    # Found assistant start, now find the end of this assistant response
                    response_start = i + T  # Start after <|im_start|>assistant\n
                    response_end = response_start
                    
                    # Search for <|im_end|> or next <|im_start|>assistant\n or end of sequence
                    while response_end < seq.numel():
                        # Check for <|im_end|>
                        if seq[response_end].item() == im_end_id:
                            response_end += 2  # Include <|im_end|> and \n in the learning
                            break
                        # Check for next <|im_start|>assistant\n (shouldn't happen in well-formed data)
                        if response_end + T <= seq.numel() and torch.equal(seq[response_end:response_end+T], seq_pat.to(seq.device)):
                            break
                        response_end += 1
                    
                    # Unmask the assistant response tokens (including <|im_end|>)
                    if response_start < response_end:
                        labels[b, response_start:response_end] = seq[response_start:response_end]
                    
                    # Move to the position after this assistant response
                    i = response_end + 1
                else:
                    i += 1
        
        # Mask padding positions in labels
        # 1. Mask positions padded to max_seq_len (padded with im_end_id)
        # 2. Mask positions padded to bd_size (padded with mask_id)
        # 3. For dummy samples (exceeded max_length), mask all labels
        for b in range(input_ids.size(0)):
            # For dummy samples, mask all labels (no loss contribution)
            if is_dummy[b]:
                labels[b, :] = self.ignore_index
                continue
            # Find positions that are padding tokens
            is_mask_pad = (input_ids[b] == mask_id)
            # Mask padding in labels
            labels[b, is_mask_pad] = self.ignore_index

        if prof_do:
            t_label_end = time.perf_counter()
            t_label = t_label_end - t_label_start

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks,
        }

        # Concatenate pixel_values along the l dimension
        if prof_do:
            t_concat_start = time.perf_counter()
        valid_pixel_values = [pv for pv in all_pixel_values if pv is not None]
        if len(valid_pixel_values) > 0:
            # Concatenate along dimension 0 (l dimension)
            batch["pixel_values"] = torch.cat(valid_pixel_values, dim=0)
        
        # Concatenate image_grid_thw
        batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

        if prof_do:
            t_concat_end = time.perf_counter()
            t_concat = t_concat_end - t_concat_start
            t_batch = t_concat_end - t_batch_start

            # Count images in this batch
            num_images = 0
            for imgs in batch_image_inputs:
                if imgs is None:
                    continue
                try:
                    num_images += len(imgs)
                except Exception:
                    num_images += 1

            max_seq = max_seq_len
            print(
                f"[DVLM_PROFILE] batch={len(all_input_ids)} images={num_images} max_seq={max_seq} "
                f"build={t_build:.4f}s proc={t_proc:.4f}s (inner={t_proc_sum:.4f}s) "
                f"labels={t_label:.4f}s concat={t_concat:.4f}s total={t_batch:.4f}s",
                flush=True,
            )
            self._profile_seen += 1

        return batch



@dataclass
class DataCollatorForMinistral(object):
    """Collate function that performs Ministral preprocessing at batch time.

    Uses tokenizer.process_messages(messages) (same as demo): chat template rendering,
    image token expansion, pixel preprocessing, and tokenization in one call.
    - Builds labels and masks tokens before assistant target sequence
    - Optionally pads to a multiple of bd_size for downstream reshape needs
    - Returns pixel_values and image_sizes (Ministral format)
    """

    tokenizer: transformers.PreTrainedTokenizer
    bd_size: Optional[int] = None
    block_causal_no_dynamic: bool = False
    max_length: Optional[int] = None
    mask_token: str = "|<MASK>|"
    ignore_index: int = IGNORE_INDEX
    target_sequence: torch.Tensor = field(default_factory=lambda: torch.tensor([10, 1503, 19464, 1010, 12, 13], dtype=torch.long)) # <|im_start|>assistant\n<think></think>

    def __call__(self, instances):
        # Filter out invalid instances
        valid_instances = []
        for ins in instances:
            messages = ins.get("messages", [])
            if not messages:
                print(f"Warning: Skipping instance with empty messages", flush=True)
                continue
            valid_instances.append(ins)

        # If all instances are invalid, create a minimal dummy batch
        if not valid_instances:
            print(f"Warning: All instances in batch are invalid, using dummy data", flush=True)
            valid_instances = [{
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello."}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]}
                ]
            }]

        # Process each instance with tokenizer.process_messages (same as demo)
        all_input_ids = []
        all_attention_masks = []
        all_pixel_values = []
        all_image_sizes = []  # Ministral uses (H, W) per image
        is_dummy = []

        im_end_id = 11
        mask_id = 100

        # Dummy WITH a tiny image so that pixel_values is always present.
        # All ranks must enter the same vision code path (real vision, line 695
        # in modeling_ministral_dlm.py); otherwise NCCL collectives desync and
        # training hangs.  A text-only dummy would set pixel_values=None and
        # route to the model-side dummy-vision branch which differs from the
        # real-vision branch, causing the hang.
        from PIL import Image as _PILImage
        _dummy_img = _PILImage.new("RGB", (32, 32), color=(0, 0, 0))
        dummy_messages = [
            {"role": "user", "content": [
                {"type": "image", "image": _dummy_img},
                {"type": "text", "text": "Hello."},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]

        def _append_vision_fields(processed):
            if "pixel_values" in processed:
                pixel_values = (
                    processed["pixel_values"] if isinstance(processed["pixel_values"], torch.Tensor)
                    else torch.tensor(processed["pixel_values"])
                )
                all_pixel_values.append(pixel_values)
                if "image_sizes" in processed:
                    sz = processed["image_sizes"]
                    all_image_sizes.append(
                        torch.tensor(sz, dtype=torch.long) if not isinstance(sz, torch.Tensor) else sz
                    )
                else:
                    n_imgs = pixel_values.shape[0] if hasattr(pixel_values, "shape") else len(pixel_values)
                    all_image_sizes.append(torch.zeros(n_imgs, 2, dtype=torch.long))
            else:
                all_pixel_values.append(None)
                all_image_sizes.append(torch.zeros(0, 2, dtype=torch.long))

        for idx, ins in enumerate(valid_instances):
            messages = ins["messages"]
            try:
                processed = self.tokenizer.process_messages(messages)
            except Exception as e:
                print(f"Warning: process_messages failed for instance {idx}: {e}, using dummy", flush=True)
                processed = self.tokenizer.process_messages(dummy_messages)
                is_dummy.append(True)
                _append_vision_fields(processed)
                input_ids_single = processed["input_ids"]
                if isinstance(input_ids_single, torch.Tensor):
                    input_ids_single = input_ids_single.squeeze(0)
                else:
                    input_ids_single = torch.tensor(input_ids_single, dtype=torch.long).squeeze(0)
                attention_mask_single = torch.ones_like(input_ids_single, dtype=torch.long)
                all_input_ids.append(input_ids_single)
                all_attention_masks.append(attention_mask_single)
                continue

            input_ids_single = processed["input_ids"]
            if isinstance(input_ids_single, torch.Tensor):
                input_ids_single = input_ids_single.squeeze(0)
                # input_ids_single = torch.cat([input_ids_single, torch.tensor([1010, 11], dtype=torch.long)], dim=0)
            else:
                input_ids_single = torch.tensor(input_ids_single, dtype=torch.long).squeeze(0)
            attention_mask_single = torch.ones_like(input_ids_single, dtype=torch.long)

            if self.max_length is not None and input_ids_single.size(0) > self.max_length:
                if 18 not in input_ids_single[self.max_length:] and 19 not in input_ids_single[self.max_length:] and 20 not in input_ids_single[self.max_length:] and 21 not in input_ids_single[self.max_length:]:
                    print(f"Warning: Sample {idx} exceeds max_length ({input_ids_single.size(0)} > {self.max_length}), but no special tokens found, using max_length", flush=True)
                    input_ids_single = input_ids_single[:self.max_length]
                    attention_mask_single = attention_mask_single[:self.max_length]
                    processed["input_ids"] = input_ids_single
                    processed["attention_mask"] = attention_mask_single
                    is_dummy.append(False)
                    _append_vision_fields(processed)
                else:
                    print(
                        f"Warning: Sample {idx} exceeds max_length ({input_ids_single.size(0)} > {self.max_length}), "
                        f"using text dummy",
                        flush=True,
                    )
                    processed = self.tokenizer.process_messages(dummy_messages)
                    input_ids_single = processed["input_ids"]
                    if isinstance(input_ids_single, torch.Tensor):
                        input_ids_single = input_ids_single.squeeze(0)
                    else:
                        input_ids_single = torch.tensor(input_ids_single, dtype=torch.long).squeeze(0)
                    attention_mask_single = torch.ones_like(input_ids_single, dtype=torch.long)
                    is_dummy.append(True)
                    _append_vision_fields(processed)
            else:
                is_dummy.append(False)
                _append_vision_fields(processed)

            # Optional pad to multiple of bd_size
            if isinstance(self.bd_size, int) and self.bd_size > 0 and self.block_causal_no_dynamic:
                seq_len = input_ids_single.size(0)
                pad_len = (self.bd_size - (seq_len % self.bd_size)) % self.bd_size
                if pad_len > 0:
                    pad_ids = torch.full((pad_len,), mask_id, dtype=input_ids_single.dtype, device=input_ids_single.device)
                    # pad_ids = torch.full((pad_len,), im_end_id, dtype=input_ids_single.dtype, device=input_ids_single.device)

                    input_ids_single = torch.cat([input_ids_single, pad_ids], dim=0)
                    pad_attn = torch.ones((pad_len,), dtype=attention_mask_single.dtype, device=attention_mask_single.device)
                    attention_mask_single = torch.cat([attention_mask_single, pad_attn], dim=0)

            all_input_ids.append(input_ids_single)
            all_attention_masks.append(attention_mask_single)

        # max_seq_len = max(ids.size(0) for ids in all_input_ids)
        # if isinstance(self.bd_size, int) and self.bd_size > 0:
        #     remainder = max_seq_len % self.bd_size
        #     if remainder != 0:
        #         max_seq_len += self.bd_size - remainder
        # batch_size = len(all_input_ids)

        # padded_input_ids = torch.full(
        #     (batch_size, max_seq_len),
        #     mask_id if mask_id is not None else im_end_id,
        #     dtype=all_input_ids[0].dtype
        # )
        # padded_attention_masks = torch.zeros((batch_size, max_seq_len), dtype=all_attention_masks[0].dtype)
        # for b in range(batch_size):
        #     seq_len = all_input_ids[b].size(0)
        #     padded_input_ids[b, :seq_len] = all_input_ids[b]
        #     padded_attention_masks[b, :seq_len] = all_attention_masks[b]

        input_ids = torch.stack(all_input_ids, dim=0)
        attention_masks = torch.stack(all_attention_masks, dim=0)
        labels = input_ids.clone()

        seq_pat = self.target_sequence # <|im_start|>assistant\n<think></think>
        T = int(seq_pat.numel())
        for b in range(input_ids.size(0)):
            seq = input_ids[b]
            labels[b, :] = self.ignore_index
            i = 0
            while i <= seq.numel() - T:
                if torch.equal(seq[i:i + T], seq_pat.to(seq.device)):
                    response_start = i + T
                    response_end = response_start
                    while response_end < seq.numel():
                        if seq[response_end].item() == im_end_id:
                            response_end += 2
                            break
                        if response_end + T <= seq.numel() and torch.equal(seq[response_end:response_end + T], seq_pat.to(seq.device)):
                            break
                        response_end += 1
                    if response_start < response_end:
                        labels[b, response_start:response_end] = seq[response_start:response_end]
                    i = response_end + 1
                else:
                    i += 1

        for b in range(input_ids.size(0)):
            if is_dummy[b]:
                labels[b, :] = self.ignore_index
                continue
            is_mask_pad = (input_ids[b] == mask_id)
            labels[b, is_mask_pad] = self.ignore_index

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks,
        }

        valid_pixel_values = [pv for pv in all_pixel_values if pv is not None]
        if len(valid_pixel_values) > 0:
            batch["pixel_values"] = torch.cat(valid_pixel_values, dim=0)
        valid_image_sizes = [s for s in all_image_sizes if s.numel() > 0]
        if len(valid_image_sizes) > 0:
            batch["image_sizes"] = torch.cat(valid_image_sizes, dim=0)
        else:
            batch["image_sizes"] = torch.zeros(0, 2, dtype=torch.long)

        return batch


class CustomMultiModalDataset(Dataset):
    """Dataset for Multi Modal data"""

    def __init__(self, dataset_path: str,
                 data_args: DatasetArguments):
        super(CustomMultiModalDataset, self).__init__()
        raw_data = json.load(open(dataset_path, "r"))
        self.data_args = data_args
        self.image_folder = data_args.image_folder

        # Normalize schema: map various image keys to 'image', convert messages->conversations,
        # drop samples without resolvable image or conversations
        normalized = []
        dropped = 0
        for item in raw_data:
            norm = self._normalize_item(item)
            if norm is not None:
                normalized.append(norm)
            else:
                dropped += 1

        self.data_dict = normalized
        print(f"Finish loading json file in dataset. Kept {len(self.data_dict)}/{len(raw_data)} after normalization, dropped {dropped} without image/conversations.")

    def __len__(self):
        return len(self.data_dict)

    def register_tokenizer(self, tokenizer, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = getattr(
            tokenizer, "image_processor", image_processor)

    def __getitem__(self, i):
        data = self.data_dict[i]
        messages = []

        # 统一将 image 字段转为列表，兼容单图（字符串）和多图（列表）
        raw_image = data['image']
        if isinstance(raw_image, list):
            image_paths = [osp.join(self.image_folder, p) for p in raw_image]
        else:
            image_paths = [osp.join(self.image_folder, raw_image)]

        image_idx = 0  # 记录下一张待使用图片的索引
        for message in data['conversations']:
            if message['from'] == 'human':
                text_value = message['value']
                content = []
                if '<image>' in text_value:
                    parts = text_value.split('<image>')
                    for idx, part in enumerate(parts):
                        if part:
                            content.append({"type": "text", "text": part})
                        if idx < len(parts) - 1:
                            # 如果这是文本中的最后一个 <image> 标签，把所有剩余的图片都放在这里
                            if idx == len(parts) - 2:
                                while image_idx < len(image_paths):
                                    content.append({"type": "image", "image": image_paths[image_idx]})
                                    image_idx += 1
                            else:
                                # 否则，每个 <image> 标签只放一张图片
                                if image_idx < len(image_paths):
                                    content.append({"type": "image", "image": image_paths[image_idx]})
                                    image_idx += 1
                else:
                    content.append({"type": "text", "text": text_value})
                    # 将所有剩余图片都添加到当前 human 消息中（适用于多视角任务）
                    while image_idx < len(image_paths):
                        content.append({"type": "image", "image": image_paths[image_idx]})
                        image_idx += 1
                messages.append({"role": "user", "content": content})
            else:
                value = message['value']
                # 2. 舍弃 <|mdm_start|> 和 <|mdm_end|>
                value = value.replace('<|mdm_start|>', '').replace('<|mdm_end|>', '')
                # 3. 将 |<NULL>| 替换为 <|NULL|>
                value = value.replace('|<NULL>|', '<|NULL|>')
                
                try:
                    data_obj = json.loads(value)
                    
                    # 递归清理字典中所有的 <|NULL|>
                    def clean_nulls(obj):
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                obj[k] = clean_nulls(v)
                        elif isinstance(obj, list):
                            for i in range(len(obj)):
                                obj[i] = clean_nulls(obj[i])
                        elif isinstance(obj, str):
                            return obj.replace('<|NULL|>', '')
                        return obj
                        
                    data_obj = clean_nulls(data_obj)
                    
                    if "explanation" in data_obj:
                        exp_text_clean = data_obj["explanation"]

                        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                            tokens = self.tokenizer.encode(exp_text_clean, add_special_tokens=False)
                            seq_len = len(tokens)
                        else:
                            seq_len = len(exp_text_clean.split())

                        block_length = getattr(self.data_args, 'block_length', 32)
                        # Fixed total budget for explanation VALUE tokens, matching
                        # inference scaffold (explanation_block_size * explanation_max_blocks).
                        # Default: 32 * 6 = 192 tokens.
                        exp_total_budget = getattr(self.data_args, 'exp_total_budget',
                                                   block_length * 6)
                        if seq_len < exp_total_budget:
                            pad_len = exp_total_budget - seq_len
                        else:
                            # Text exceeds budget — round up to next multiple of block_length
                            pad_len = (block_length - (seq_len % block_length)) % block_length
                            if pad_len == 0:
                                pad_len = block_length
                        data_obj["explanation"] = exp_text_clean + "<|NULL|>" * pad_len

                    # 4. FMB: pad longitudinal/lateral values to fixed 3 tokens with <|NULL|>
                    #    so scaffold structure is consistent across samples.
                    if "future_meta_behavior" in data_obj and isinstance(data_obj["future_meta_behavior"], dict):
                        fmb = data_obj["future_meta_behavior"]
                        fmb_budget = 3  # max token length among all FMB values
                        for fmb_key in ["longitudinal", "lateral"]:
                            if fmb_key in fmb and isinstance(fmb[fmb_key], str):
                                val = fmb[fmb_key]
                                if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                                    val_toks = self.tokenizer.encode(val, add_special_tokens=False)
                                    pad_needed = fmb_budget - len(val_toks)
                                    if pad_needed > 0:
                                        fmb[fmb_key] = val + "<|NULL|>" * pad_needed
                        data_obj["future_meta_behavior"] = fmb

                    # 5. Trajectory: normalize coordinates to fixed +XXX.XX format
                    #    and insert spaces to prevent BPE merging.
                    #    例: [[+14.70,-00.04], [+108.02,-02.97]]
                    #      → [[ +014.70, -000.04], [ +108.02, -002.97]]
                    if "trajectory" in data_obj and isinstance(data_obj["trajectory"], str):
                        import re
                        traj = data_obj["trajectory"]
                        def _fmt_coord(m):
                            """Reformat coordinate to +XXX.XX (3 int digits, 2 decimal)."""
                            raw = m.group(0)
                            sign = raw[0]
                            num = float(raw[1:])
                            return f"{sign}{num:06.2f}"
                        # Normalize each coordinate: +14.70 → +014.70
                        traj = re.sub(r'[+-]\d+\.\d+', _fmt_coord, traj)
                        # Insert spaces: ,+ → , +  and  [+ → [ +
                        traj = re.sub(r',([+-])', r', \1', traj)
                        traj = re.sub(r'\[([+-])', r'[ \1', traj)
                        data_obj["trajectory"] = traj

                    value = json.dumps(data_obj, ensure_ascii=False)
                except Exception:
                    pass

                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": value},
                        ],
                    }
                )
        # 返回 messages + 原始 metadata（供 reward 函数使用）
        result = {"messages": messages}
        # Carry over metadata fields (e.g., future waypoints, sample_id)
        for k, v in data.items():
            if k not in ("image", "conversations"):
                result[k] = v
        return result

    def _normalize_item(self, item):
        # Resolve image field from multiple possible keys.
        # 支持单图（字符串）和多图（列表）两种形式：
        #   单图: "image": "path/to/img.jpg"
        #   多图: "image": ["path1.jpg", "path2.jpg"]
        #         "images": ["path1.jpg", "path2.jpg"]
        #         "image_paths": ["path1.jpg", "path2.jpg"]
        image_value = item.get('image')
        if not image_value:
            image_path = item.get('image_path')
            image_paths = item.get('image_paths')
            images = item.get('images')
            if isinstance(image_path, str) and image_path:
                image_value = image_path
            elif isinstance(image_paths, list) and len(image_paths) > 0 and isinstance(image_paths[0], str):
                image_value = image_paths        # ← 保留完整列表
            elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], str):
                image_value = images             # ← 保留完整列表
        elif isinstance(image_value, list):
            # "image" 字段本身就是列表，直接保留
            image_value = [p for p in image_value if isinstance(p, str) and p]

        # Resolve conversations/messages array
        conv = item.get('conversations')
        if conv is None:
            conv = item.get('messages')

        if not image_value or not isinstance(conv, list) or len(conv) == 0:
            return None

        # Normalize each message to {from, value}
        norm_conv = []
        for m in conv:
            frm = m.get('from')
            if not frm:
                role = m.get('role')
                if role in ('user', 'human'):
                    frm = 'human'
                elif role in ('assistant', 'gpt', 'bot'):
                    frm = 'gpt'
                else:
                    frm = role
            value = m.get('value')
            if value is None:
                value = m.get('text') or m.get('content') or ''
            norm_conv.append({'from': frm, 'value': value})

        if len(norm_conv) == 0:
            return None

        result = {'image': image_value, 'conversations': norm_conv}
        # Preserve metadata fields (e.g., future waypoints, sample_id) for reward computation
        for k, v in item.items():
            if k not in ('image', 'conversations', 'messages', 'image_path', 'image_paths', 'images'):
                result[k] = v
        return result



def preprocess_multimodal_llava(sources, data_args):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(
                                DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.use_image_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + \
                    replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token)
    return sources


def tokenizer_image_token(prompt,
                          tokenizer,
                          image_token_index=IMAGE_TOKEN_INDEX,
                          return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def preprocess_llama_from_llava_plain(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False):
    """
    This function just add the image in the front of text.
    And don't add any prompt.
    Args:
        sources: The input data with text and image.
        tokenizer: The tokenizer to process text.
        has_image: Whether the input data has image.
    Returns:
        The input_ids and labels for the model.
    """
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess_llama_from_llava_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False):
    """
    This function add the prompt and then put the image after the prompt.
    So it needs additional code to generate the target label.
    Args:
        sources: The input data with text and image.
        tokenizer: The tokenizer to process text.
        has_image: Whether the input data has image.
    Returns:
        The input_ids and labels for the model.
    """

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # squeeze possible leading batch dim from dataset __getitem__ (e.g., [1, T])
        def _squeeze_leading_dim(x):
            if isinstance(x, torch.Tensor) and x.dim() > 1 and x.size(0) == 1:
                return x.squeeze(0)
            return x
        input_ids = [_squeeze_leading_dim(x) for x in input_ids]
        labels = [_squeeze_leading_dim(x) for x in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'pixel_values' in instances[0]:
            pixel_values = [instance['pixel_values'] for instance in instances]
            if all(x is not None and x.shape == pixel_values[0].shape for x in pixel_values):
                batch['pixel_values'] = torch.stack(pixel_values)
            else:
                batch['pixel_values'] = pixel_values
        # pass through auxiliary vision metadata if present (e.g., image_grid_thw)
        if 'image_grid_thw' in instances[0]:
            image_grid_thw = [instance['image_grid_thw'] for instance in instances]
            try:
                batch['image_grid_thw'] = torch.stack([
                    torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in image_grid_thw
                ])
            except Exception:
                batch['image_grid_thw'] = image_grid_thw
        return batch
