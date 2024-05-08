import torch
import copy
import megfile
import random
import webdataset as wds
import ipdb

from loguru import logger
from typing import List, Optional, Tuple, Union, Dict

from .base_dataset import BaseDataset
from mmgpt.utils.constants import *
from mmgpt.utils.mm_utils import filter_no_caption_or_no_image
from mmgpt.utils.dist_utils import get_world_size
    

class PairWebDataset(BaseDataset):
    """Interleave format dataset stage1 pre-training."""

    def __init__(self, dataset, tokenizer, multimodal_cfg):
        BaseDataset.__init__(self, dataset, tokenizer, multimodal_cfg)

        if isinstance(PAIR_WEBDATA[dataset]['path'], list):
            shards_list = PAIR_WEBDATA[dataset]['path']
        else:
            shards_list = megfile.smart_glob(PAIR_WEBDATA[dataset]['path'])

        data_size = PAIR_WEBDATA[dataset]['size']
        merge_round = PAIR_WEBDATA[dataset]['merge_round']

        if 'laion2b' in dataset:
            dataformat = ("jpg;png;jpeg", "json")
        elif 'taisu' in dataset:
            dataformat = ("jpg;png;jpeg", "txt")
        else:
            dataformat = ("jpg;png;jpeg", "txt", "json")

        dp = wds.DataPipeline(
            # wds.DistributedSimpleShardList(shards_list, seed=3407),
            wds.InfiniteShardList(shards_list, seed=3407),
            wds.shuffle(get_world_size(), handler=wds.warn_and_continue, rng=random.Random(42)),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue, rng=random.Random(42)),
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple(*dataformat, handler=wds.warn_and_continue),
        )

        self.data_size = data_size
        self.merge_round = merge_round
        self.use_im_start_end = self.multimodal_cfg['use_im_start_end']
        self.inner_iter = iter(dp)
        self.replace_token = DEFAULT_IM_PATCH_TOKEN * self.multimodal_cfg['image_token_len']
        if self.use_im_start_end:
            self.replace_token = DEFAULT_IM_START_TOKEN + self.replace_token + DEFAULT_IM_END_TOKEN

        logger.info(f"{data_size // self.merge_round} interleaved ({self.merge_round}-merged) image-text pairs (splitted to {get_world_size()} workers) are sampled from dataset: {dataset}.")

    def add_image_token(self, text):
        if DEFAULT_IMAGE_TOKEN in text:
            text = text.replace(DEFAULT_IMAGE_TOKEN, self.replace_token)
        else:
            text = self.replace_token + '\n' + text
        return text

    def token_processor(self, text_list, image_list):
        input_ids, targets = [], []
        for i, (prompt, text) in enumerate(text_list):
            # ipdb.set_trace()
            # in each text list, we **may** have a task prompt and **must** have a text
            # if we have a task prompt, we make sure that it contains else we must manually add one image token
            # if we don't have any task prompts, we also make sure that it contains else we must manually add one image token
            if prompt is not None:
                prompt = self.add_image_token(prompt)
            else:
                text = self.add_image_token(text)

            # we individually tokenize prompt (if have) and text tokens (with added eos)
            # for more conviniently mask prompt tokens during training
            prompt_input_ids = self.tokenizer(
                prompt,
                padding="longest",
                # note that we must claim max_length here to avoid the warning of "Token indices sequence length is longer than ..."
                # in fact, prompt text is always composed of less than ten words.
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids if prompt is not None else []
            text_input_ids = self.tokenizer(
                text + self.tokenizer.eos_token,
                padding="longest",
                # we make sure here that each prompt + text pair maintains a max token length which is smaller than max length
                max_length=self.tokenizer.model_max_length - len(prompt_input_ids),
                truncation=True,
            ).input_ids

            # to avoid duplicated bos (if have, e.g. baichuan, phi-series has no bos), manually remove the bos if prompt exists
            if prompt is not None and text_input_ids[0] == 1:
                text_input_ids = text_input_ids[1:]

            # in advance we check the possibility of the length of tokens exceeding the model max context length
            # certainly, we already make sure that the max token length of all image-text pairs is smaller than max length
            # so that in each training step we at least train one image & text pair
            if len(input_ids) + len(prompt_input_ids) + len(text_input_ids) > self.tokenizer.model_max_length:
                # remember to truncate image here so we need not to care truncated image tokens any more!
                image_list = image_list[:i]
                logger.info(f'exceeding max length {self.tokenizer.model_max_length}, ignore last {len(text_list) - i} samples!')
                logger.info(text_list[-1])
                break

            # merge the potential prompt_input_ids and text input ids
            # also, mask all prompt and mask all image tokens in text
            input_ids.extend(prompt_input_ids + text_input_ids)
            targets.extend([IGNORE_INDEX] * len(prompt_input_ids) + text_input_ids)

        # # for megatron-like architexture, we need to pad tokens to max length in advance
        # if len(input_ids) < self.tokenizer.model_max_length:
        #     num_padded_tokens = self.tokenizer.pad_token_id - len(input_ids)
        #     input_ids += [self.tokenizer.pad_token_id] * num_padded_tokens
        #     targets += [IGNORE_INDEX] * num_padded_tokens

        # tensorize interleaved prompted texts and targets
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)

        # constructing additional image token related mask indices (doing on targets):
        # 1. image patch tokens are masked for image understanding
        image_indices = (targets == self.im_patch_token)

        # 2. image start and end token
        # this is optional, since in Llava-1.0, both tokens are unmasked
        if self.use_im_start_end:
            image_indices = torch.logical_or(
                image_indices,
                torch.logical_or(
                    targets == self.im_start_token, 
                    targets == self.im_end_token
                )
            )

        # construct image masks from input_ids and apply masking 
        targets = targets.masked_fill(image_indices, IGNORE_INDEX)
        
        # avoid empty image list, for double checking
        if len(image_list) == 0:
            image_list = [torch.zeros(3, self.image_size, self.image_size)]

        return dict(
            image=image_list,
            input_ids=input_ids, 
            labels=targets, 
        )

    def __len__(self):
        return self.data_size // self.merge_round
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        text_list = []
        image_list = []

        for _ in range(self.merge_round):
            data = next(self.inner_iter)

            if len(data) == 3:
                image, text, info = data
                task_prompt = info['task_prompt'] if 'task_prompt' in info else None
            else:
                image, info = data
                if isinstance(info, dict):
                    text = str(info['caption'])
                    task_prompt = info['task_prompt'] if 'task_prompt' in info else None
                else:
                    text = info
                    task_prompt = None    

            # assume that all fail is impossible
            try:
                image = self.image_processor(copy.deepcopy(image))
                image_list.append(image)
            except Exception as e:
                logger.exception(e)
                continue

            # maybe need not to deepcopy? use for train safely
            text_list.append(copy.deepcopy((task_prompt, text)))

        return self.token_processor(text_list, image_list)