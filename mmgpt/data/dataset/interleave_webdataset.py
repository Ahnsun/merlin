import os
import torch
import copy
import random
import megfile
import webdataset as wds

from loguru import logger
from typing import List, Optional, Tuple, Union, Dict

from .base_dataset import BaseDataset
from mmgpt.utils.constants import *
from mmgpt.utils.mm_utils import filter_no_text_or_no_image
from mmgpt.utils.dist_utils import get_world_size
    

class InterleaveWebDataset(BaseDataset):
    """Interleave format dataset stage1 pre-training."""

    def __init__(self, dataset, tokenizer, multimodal_cfg):
        BaseDataset.__init__(self, dataset, tokenizer, multimodal_cfg)

        if isinstance(INTERLEAVE_WEBDATA[dataset]['path'], list):
            shards_list = INTERLEAVE_WEBDATA[dataset]['path']
        else:
            shards_list = megfile.smart_glob(INTERLEAVE_WEBDATA[dataset]['path'])

        data_size = INTERLEAVE_WEBDATA[dataset]['size']

        dp = wds.DataPipeline(
            # wds.DistributedSimpleShardList(shards_list, seed=3407),
            wds.InfiniteShardList(shards_list, seed=3407),
            wds.shuffle(get_world_size(), handler=wds.warn_and_continue, rng=random.Random(42)),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue, rng=random.Random(42)),
            wds.select(filter_no_text_or_no_image),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_dict("jpg;png;jpeg", "txt;json", handler=wds.warn_and_continue),
        )

        self.data_size = data_size
        self.inner_iter = iter(dp)
        logger.info(f"{data_size} interleaved image-texts (splitted to {get_world_size()} workers) are sampled from dataset: {dataset}.")

    def multimodal_processor(self, text_list, image_text_index_list):
        new_text_list = [copy.deepcopy(text) for text in text_list]

        if len(image_text_index_list) == 0:
            pass
        elif image_text_index_list[-1] == len(new_text_list):
            new_text_list.append('')
            # image_text_index_list = [idx - 1 for idx in image_text_index_list]
        elif image_text_index_list[-1] > len(new_text_list):
            while image_text_index_list[-1] > len(new_text_list):
                logger.warning('drop one last out of boundary image')
                image_text_index_list = image_text_index_list[:-1]

        for idx in image_text_index_list:
            new_text_list[idx] = DEFAULT_IMAGE_TOKEN + '\n' + new_text_list[idx]

        text = " ".join(new_text_list) + self.tokenizer.eos_token

        replace_token = DEFAULT_IM_PATCH_TOKEN * self.multimodal_cfg['image_token_len']
        if self.multimodal_cfg['use_im_start_end']:
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        text = text.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        return text

    def token_processor(self, texts):
        # Tokenize interleaved texts
        input_ids = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()

        targets = targets.masked_fill(
            torch.logical_or(
                targets == self.tokenizer.pad_token_id, 
                targets == self.im_patch_token
            ), IGNORE_INDEX
        )

        if self.multimodal_cfg['use_im_start_end']:
            targets = targets.masked_fill(
                torch.logical_or(
                    targets == self.im_start_token, 
                    targets == self.im_end_token
                ), IGNORE_INDEX
            )
            
        return dict(
            input_ids=input_ids,
            labels=targets,
        )

    def __len__(self):
        return self.data_size

    def to_dict(self, sample):
        sample_info = sample["json"]
        
        try:
            text_list = sample_info['text_list']
        except:
            logger.info(sample_info)
            text_list = []

        try:
            image_infos = sample_info['image_info']
        except:
            logger.info(sample_info)
            image_infos = []

        image_list, image_text_index_list = [], []
        for image_info in image_infos:
            try:
                image_file = image_info["image_name"]

                if 'matched_sim' in image_info:
                    matched_sim = image_info["matched_sim"]
                elif 'match_sim' in image_info:
                    matched_sim = image_info["match_sim"]
                else:
                    # default by 1
                    matched_sim = 1

                # filter image by clip similarity = 0.25
                # since mmc4 has been already filtered, and chinese clip scores are typically high.
                if matched_sim < 0.25:
                    continue

                # transfer suffix of image to jpg
                if '.' in image_file:
                    image_file = image_file.split('.')[0] + '.jpg'
                else:
                    image_file += '.jpg'

                # 000000-0.jpg
                if image_file in sample:
                    image = sample[image_file]
                # 0.jpg
                elif image_file.split('-')[-1] in sample:
                    image = sample[image_file.split('-')[-1]]
                else:
                    logger.warning(f"wrong type of image!")

                image = self.image_processor(copy.deepcopy(image))
                image_list.append(image)

                index = image_info["matched_text_index"]
                image_text_index_list.append(index)
            except Exception as e:
                logger.exception(e)
                logger.info(sample)
                logger.warning(f"Image not found: {len(image_infos)}, {image_file}")

        text = self.multimodal_processor(text_list, image_text_index_list)

        # align with fastchat & llava here, put the conversation into a list for tokenization
        data_dict = self.token_processor([text])
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # check out if appended image token exceeds the maximum length
        images_left = torch.where(data_dict["input_ids"] == self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN])[0])[0]
        num_images_right = 0
        if images_left.shape[0] > 0 and len(image_list) > 0:
            images_right = images_left + self.multimodal_cfg['image_token_len'] + 1
            num_images_right = torch.where(images_right < data_dict["input_ids"].shape[0])[0].shape[0]
            if num_images_right < images_left.shape[0]:
                data_dict["input_ids"] = torch.cat([data_dict["input_ids"][:images_left[num_images_right]], torch.tensor([self.tokenizer.eos_token_id])])
                data_dict["labels"] = torch.cat([data_dict["labels"][:images_left[num_images_right]], torch.tensor([self.tokenizer.eos_token_id])])

        if num_images_right > 0 and len(image_list) > 0:
            data_dict['image'] = image_list[:num_images_right]
        else:
            data_dict['image'] = [torch.zeros(3, self.image_size, self.image_size)]

        return data_dict

    def __getitem__(self, index):
        return self.to_dict(next(self.inner_iter))


