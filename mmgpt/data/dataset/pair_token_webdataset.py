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
    

class PairTokenWebDataset(BaseDataset):
    """Interleave format dataset stage1 pre-training."""

    def __init__(self, dataset, tokenizer, multimodal_cfg):
        BaseDataset.__init__(self, dataset, tokenizer, multimodal_cfg)

        if isinstance(PAIR_WEBDATA[dataset]['path'], list):
            shards_list = PAIR_WEBDATA[dataset]['path']
        else:
            shards_list = megfile.smart_glob(PAIR_WEBDATA[dataset]['path'])

        data_size = PAIR_WEBDATA[dataset]['size']
        merge_round = PAIR_WEBDATA[dataset]['merge_round']

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
            wds.to_tuple("jpg;png;jpeg", "txt", "json", handler=wds.warn_and_continue),
        )

        self.data_size = data_size
        self.merge_round = merge_round
        self.inner_iter = iter(dp)

        logger.info(f"{data_size // self.merge_round} interleaved ({self.merge_round}-merged) image-text pairs (splitted to {get_world_size()} workers) are sampled from dataset: {dataset}.")

    def token_processor(self, token_list, text_list, image_list):
        input_ids, targets = [], []
        for i, (input_id, target) in enumerate(token_list):
            # we check the possibility of the length of tokens exceeding the model max context length
            # certainly, we already make sure that the max token length of all image-text pairs is smaller than max length
            if len(input_id) > self.tokenizer.model_max_length:
                input_id = input_id[:self.tokenizer.model_max_length - 1] + [2]
                target = target[:self.tokenizer.model_max_length - 1] + [1]

            # so that in each training step we at least train one image & text pair
            if len(input_ids) + len(input_id) > self.tokenizer.model_max_length:
                # remember to truncate image here so we need not to care truncated image tokens any more!
                image_list = image_list[:i]
                logger.info(f'exceeding max length {self.tokenizer.model_max_length}, ignore last {len(text_list) - i} samples!')
                logger.info(text_list[-1])
                break

            # merge the potential prompt_input_ids and text input ids
            # also, mask all prompt and mask all image tokens in text
            input_ids.extend(input_id)
            targets.extend(target)

        # # for megatron-like architexture, we need to pad tokens to max length in advance
        # if len(input_ids) < self.tokenizer.model_max_length:
        #     num_padded_tokens = self.tokenizer.pad_token_id - len(input_ids)
        #     input_ids += [self.tokenizer.pad_token_id] * num_padded_tokens
        #     targets += [IGNORE_INDEX] * num_padded_tokens

        # tensorize interleaved prompted texts and targets
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)

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
        token_list = []
        text_list = []
        image_list = []

        for _ in range(self.merge_round):
            image, text, info = next(self.inner_iter)

            # assume that all fail is impossible
            try:
                image = self.image_processor(copy.deepcopy(image))
                image_list.append(image)
            except Exception as e:
                logger.exception(e)
                continue

            text_list.append(copy.deepcopy(text))
            token_list.append(copy.deepcopy((info['input_ids'], info['labels'])))

        return self.token_processor(token_list, text_list, image_list)