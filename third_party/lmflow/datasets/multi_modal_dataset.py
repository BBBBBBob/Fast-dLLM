#!/usr/bin/env python
# coding=utf-8
# FIXME update the doc string.
"""This Python code defines a class Multi Modal Dataset.
"""
import copy
from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import os.path as osp
import transformers
import torch
from torch.utils.data import Dataset

try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # optional until Qwen VL training stack is used
    process_vision_info = None  # type: ignore[misc, assignment]

from lmflow.args import DatasetArguments
from lmflow.utils import llava_conversation_lib as conversation_lib

from lmflow.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def _llava_conversations_to_qwen_messages(
    conversations: List[Dict[str, Any]],
    pil_image: Optional[Image.Image],
) -> List[Dict[str, Any]]:
    """Map LLaVA ``human`` / ``gpt`` turns to Qwen ``messages`` (first user turn may include a PIL image)."""
    role_from = {"human": "user", "gpt": "assistant"}
    messages: List[Dict[str, Any]] = []
    pending_image = pil_image is not None
    for turn in conversations:
        role = role_from.get(turn.get("from"), "user")
        val = turn.get("value", "")
        if role == "user" and pending_image:
            text = (
                val.replace(DEFAULT_IMAGE_TOKEN, "")
                .replace("<image>", "")
                .strip()
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": text},
                    ],
                }
            )
            pending_image = False
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": val}]})
    return messages


class CustomMultiModalDataset(Dataset):
    """LMFlow ``custom_multi_modal`` dataset (JSON list).

    ``dataset_path`` must be a single JSON file whose root is a **list** of samples.
    Each sample uses ``image`` (filename under ``image_folder``) and ``conversations``
    (``human`` / ``gpt`` turns), e.g.::

        {
            "image": "example.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\\nWhat is in the picture?"},
                {"from": "gpt", "value": "A cat."}
            ]
        }

    See LMFlow documentation for multimodal / image-text dataset formats.

    Set ``data_args.return_as_qwen_messages=True`` (see :class:`lmflow.args.MultiModalDatasetArguments`)
    to return ``{"messages": [...]}`` only, for Qwen ``processor`` + :class:`DataCollatorForQwenVL`.
    In that mode images are raw RGB PIL; resizing / padding is done in the collator via ``processor``.
    """

    def __init__(self, dataset_path: str,
                 data_args: DatasetArguments):
        super(CustomMultiModalDataset, self).__init__()
        data_dict = json.load(open(dataset_path, "r"))
        self.data_dict = data_dict
        print("Finish loading json file in dataset.")
        self.data_args = data_args
        self.image_folder = data_args.image_folder

    def __len__(self):
        return len(self.data_dict)

    def register_tokenizer(self, tokenizer, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = getattr(
            tokenizer, "image_processor", image_processor)

    def __getitem__(self, i):
        data = self.data_dict[i]
        if isinstance(i, int):
            data = [data]
        assert len(data) == 1
        sample0 = data[0]

        if getattr(self.data_args, "return_as_qwen_messages", False):
            pil_image: Optional[Image.Image] = None
            if "image" in sample0:
                image_file = sample0["image"]
                pil_image = Image.open(
                    osp.join(self.image_folder, image_file),
                ).convert("RGB")
            messages = _llava_conversations_to_qwen_messages(
                sample0["conversations"],
                pil_image,
            )
            return {"messages": messages}

        processor = self.image_processor
        if 'image' in data[0]:
            image_file = data[0]['image']
            image = Image.open(
                osp.join(self.image_folder, image_file)).convert("RGB")
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result    
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            data = preprocess_multimodal_llava(
                copy.deepcopy([e["conversations"] for e in data]),
                self.data_args)
        else:
            data = copy.deepcopy([e["conversations"] for e in data])
        if self.data_args.sep_style == "plain":
            data_dict = preprocess_llama_from_llava_plain(
                data,
                self.tokenizer,
                has_image=('image' in self.data_dict[i])
            )
        else:
            data_dict = preprocess_llama_from_llava_v1(
                data,
                self.tokenizer,
                has_image=('image' in self.data_dict[i])
            )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        
        # image exist in the data
        if 'image' in self.data_dict[i]:
            data_dict['image'] = image
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.image_processor.crop_size
            data_dict['image'] = torch.zeros(
                3, crop_size['height'], crop_size['width'])
        return data_dict



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


def _vision_info_for_messages(messages: List[Dict[str, Any]]) -> Tuple[Any, Any, Optional[Dict[str, Any]], Optional[List[Any]]]:
    """Call ``process_vision_info``; support 2- or 3-tuple returns and optional (video, meta) pairs."""
    if process_vision_info is None:
        raise ImportError("Install qwen-vl-utils for DataCollatorForQwenVL (pip install qwen-vl-utils).")
    try:
        out = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
    except TypeError:
        out = process_vision_info(messages)
    video_kwargs: Optional[Dict[str, Any]] = None
    video_metadatas: Optional[List[Any]] = None
    if len(out) == 2:
        image_inputs, video_inputs = out[0], out[1]
    elif len(out) == 3:
        image_inputs, video_inputs, video_kwargs = out[0], out[1], out[2]
        if video_inputs is not None and len(video_inputs) > 0:
            first = video_inputs[0]
            if isinstance(first, (tuple, list)) and len(first) == 2:
                vids, metas = zip(*video_inputs)
                video_inputs, video_metadatas = list(vids), list(metas)
    else:
        raise ValueError(f"process_vision_info returned length {len(out)}, expected 2 or 3")
    return image_inputs, video_inputs, video_kwargs, video_metadatas


@dataclass
class DataCollatorForQwenVL(object):
    """Qwen-VL batching for ``{"messages": ...}`` rows from :class:`CustomMultiModalDataset`.

    Per sample: ``processor.apply_chat_template`` → ``process_vision_info`` →
    ``processor(text=[...], images=..., videos=..., ...)`` (same as ``fast_dvlm/run_chatbot.py``),
    then pads ``input_ids`` / ``labels`` and masks non-assistant tokens in ``labels``.
    """

    processor: Any
    tokenizer: transformers.PreTrainedTokenizer
    ignore_index: int = IGNORE_INDEX
    target_sequence: torch.Tensor = field(
        default_factory=lambda: torch.tensor([151644, 77091, 198], dtype=torch.long),
    )

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_input_ids: List[torch.Tensor] = []
        all_pixel_values: List[Optional[torch.Tensor]] = []
        all_image_grid_thw: List[torch.Tensor] = []
        all_pixel_values_videos: List[Optional[torch.Tensor]] = []
        all_video_grid_thw: List[Optional[torch.Tensor]] = []
        all_second_per_grid_ts: List[Optional[torch.Tensor]] = []

        for ins in instances:
            messages = ins["messages"]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            image_inputs, video_inputs, video_kwargs, video_metadatas = _vision_info_for_messages(
                messages,
            )

            proc_kwargs: Dict[str, Any] = {
                "text": [text],
                "padding": False,
                "return_tensors": "pt",
            }
            if image_inputs is not None and len(image_inputs) > 0:
                proc_kwargs["images"] = image_inputs
            if video_inputs is not None and len(video_inputs) > 0:
                proc_kwargs["videos"] = video_inputs
            if video_kwargs:
                proc_kwargs.update(video_kwargs)
            if video_metadatas is not None and len(video_metadatas) > 0:
                proc_kwargs["video_metadata"] = video_metadatas
                proc_kwargs.setdefault("do_resize", False)

            inputs = self.processor(**proc_kwargs)
            input_ids_single = inputs["input_ids"].squeeze(0)

            if "pixel_values" in inputs:
                all_pixel_values.append(inputs["pixel_values"])
            else:
                all_pixel_values.append(None)
            if "image_grid_thw" in inputs:
                all_image_grid_thw.append(inputs["image_grid_thw"])
            else:
                all_image_grid_thw.append(torch.tensor([[0, 0, 0]], dtype=torch.long))
            all_pixel_values_videos.append(
                inputs["pixel_values_videos"] if "pixel_values_videos" in inputs else None,
            )
            all_video_grid_thw.append(
                inputs["video_grid_thw"] if "video_grid_thw" in inputs else None,
            )
            all_second_per_grid_ts.append(
                inputs["second_per_grid_ts"] if "second_per_grid_ts" in inputs else None,
            )

            all_input_ids.append(input_ids_single)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id)
        labels = input_ids.clone()

        im_end_token = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        im_end_id = im_end_token[0] if len(im_end_token) == 1 else 151645

        seq_pat = self.target_sequence
        T = int(seq_pat.numel())
        for b in range(input_ids.size(0)):
            seq = input_ids[b]
            labels[b, :] = self.ignore_index
            idx = 0
            while idx <= seq.numel() - T:
                if torch.equal(seq[idx : idx + T], seq_pat.to(seq.device)):
                    response_start = idx + T
                    response_end = response_start
                    while response_end < seq.numel():
                        if seq[response_end].item() == im_end_id:
                            response_end += 2
                            break
                        if (
                            response_end + T <= seq.numel()
                            and torch.equal(
                                seq[response_end : response_end + T],
                                seq_pat.to(seq.device),
                            )
                        ):
                            break
                        response_end += 1
                    if response_start < response_end:
                        labels[b, response_start:response_end] = seq[response_start:response_end]
                    idx = response_end + 1
                else:
                    idx += 1

        labels = labels.masked_fill(input_ids == self.tokenizer.pad_token_id, self.ignore_index)

        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks,
        }
        valid_pv = [pv for pv in all_pixel_values if pv is not None]
        if len(valid_pv) > 0:
            batch["pixel_values"] = torch.cat(valid_pv, dim=0)
        batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
        valid_pvv = [x for x in all_pixel_values_videos if x is not None]
        if len(valid_pvv) > 0:
            batch["pixel_values_videos"] = torch.cat(valid_pvv, dim=0)
        valid_vg = [x for x in all_video_grid_thw if x is not None]
        if len(valid_vg) > 0:
            batch["video_grid_thw"] = torch.cat(valid_vg, dim=0)
        valid_spg = [x for x in all_second_per_grid_ts if x is not None]
        if len(valid_spg) > 0:
            batch["second_per_grid_ts"] = torch.cat(valid_spg, dim=0)
        return batch


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
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

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        return batch