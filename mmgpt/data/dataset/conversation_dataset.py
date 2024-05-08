import os
import copy
import torch
import megfile

from loguru import logger
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .base_dataset import BaseDataset
from mmgpt.utils.constants import *
import mmgpt.utils.conversation as conversation_lib


class ConversationDataset(BaseDataset):
    def __init__(self, datasets, tokenizer, multimodal_cfg):
        super(ConversationDataset, self).__init__(datasets, tokenizer, multimodal_cfg)
        
        list_data_dict = []
        list_image_path = []
        for name in datasets.split("+"):
            dataset = CONVERSATION_DATA[name]

            data_path = dataset['annotations']
            if data_path.endswith('.json'):
                data = self.load_json(data_path)
            else:
                paths = megfile.smart_glob(os.path.join(data_path, "*.json"))
                data = self.parallel_load_json(json_paths=paths)

            if 'merge_round' in dataset:
                data = self.merge_conversations(data, dataset['merge_round'])

            data_freq = dataset['frequency']
            if isinstance(data_freq, int):
                sampled_data = data * data_freq
            else:
                if 'cn' in name:
                    sampled_data = data[-int(len(data) * data_freq):]
                else:
                    sampled_data = data[:int(len(data) * data_freq)]
            logger.info(f"Data from {data_path} are sampled from {len(data)} to {len(sampled_data)} conversations (x{data_freq}).")
            list_data_dict.extend(sampled_data)

            image_path = dataset['images']
            list_image_path.extend([image_path] * len(sampled_data))

        assert len(list_data_dict) == len(list_image_path)
        logger.info(f"{len(list_data_dict)} conversations in total.")

        self.list_data_dict = list_data_dict
        self.list_image_path = list_image_path
    
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if ('image' in sample or 'images' in sample) else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if ('image' in sample or 'images' in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def multimodal_processor(self, conversations, box_text_list):
        idx = 0
        for sentence in conversations:
            replace_token = DEFAULT_IM_PATCH_TOKEN * self.multimodal_cfg['image_token_len']
            if self.multimodal_cfg['use_im_start_end']:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            box_count = sentence["value"].count(DEFAULT_BOX_TOKEN)
            if box_count > 0 and box_text_list is not None:
                for i in range(box_count):
                    sentence["value"] = sentence["value"].replace(DEFAULT_BOX_TOKEN, box_text_list[idx + i], 1)
                idx += box_count
        return conversations

    def token_processor(self, sources):
        conv = conversation_lib.conv.copy()
        roles = {
            "human": conv.roles[0], 
            "assistant": conv.roles[1], 
            "gpt": conv.roles[1],
            "obj365": conv.roles[1],
            "vg": conv.roles[1],
        }

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"].lower()] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"].lower()]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)

            if 'baichuan' in self.tokenizer.name_or_path.lower():
                cur_len = 0
                target[:cur_len] = IGNORE_INDEX # keep bos
                for i, rou in enumerate(rounds):
                    if rou == "":
                        break
                    round_len = len(self.tokenizer(rou + conv.sep2).input_ids)

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    instruction_len = len(self.tokenizer(parts[0]).input_ids) - 1

                    # Ignore the user instructions
                    target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                    cur_len += round_len
            else:
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX # keep bos
                for i, rou in enumerate(rounds):
                    if rou == "":
                        break
                    round_len = len(self.tokenizer(rou).input_ids)

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
                    instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

                    # Ignore the user instructions
                    target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                    cur_len += round_len

            target[cur_len:] = IGNORE_INDEX

            if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    logger.exception(
                        f"info: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(input_ids=input_ids, labels=targets)

    def __getitem__(self, i):
        data = copy.deepcopy(self.list_data_dict[i])
        
        # Deal with various types of conversation data.
        if isinstance(data, dict):
            conversations = data["conversations"]
        else:
            conversations = data

        # Deal with multimodal and language-only data simultaneously.
        if isinstance(data, dict) and ('image' in data.keys() or 'images' in data.keys()):
            box_text_list = None
            image_list, image_wh_list = [], []
            if 'image' in data.keys():
                image_path = self.list_image_path[i] + data['image']
                try:
                    image = self.load_images(image_path)
                    image_wh_list.append(image.size)
                    image = self.image_processor(image)
                    image_list.append(image)
                except Exception as e:
                    logger.exception(e)
                    logger.exception(f'image {image_path} are broken or grayscale! we thus use zero-image instead!')

            else:
                for img in data['images']:
                    image_path = self.list_image_path[i] + img
                    try:
                        image = self.load_images(image_path)
                        image_wh_list.append(image.size)
                        image = self.image_processor(image)
                        image_list.append(image)
                    except Exception as e:
                        logger.exception(e)
                        logger.exception(f'image {image_path} are broken or grayscale! we thus use zero-image instead!')

            insert_img = 0
            for c in conversations:
                if DEFAULT_BOX_TOKEN in c['value']:
                    insert_img = 1
                    break

            if len(image_list) > 0 and 'boxes' in data and insert_img:
                # (DEFAULT_BOX_TOKEN in conversations[0]['value'] or DEFAULT_BOX_TOKEN in conversations[1]['value']):
                boxes = data['boxes']
                boxes, conversations = self.box_shuffle_and_sample(boxes, conversations)
                box_text_list = self.box_processor([boxes], image_wh_list, image_path)

            conversations = [self.multimodal_processor(conversations, box_text_list)]
        else:
            conversations = [conversations]

        # align with fastchat & llava here, put the conversation into a list for tokenization
        data_dict = self.token_processor(conversations)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # check out if appended image token exceeds the maximum length
        images_left = torch.where(data_dict["input_ids"] == self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN])[0])[0]
        num_images_right = 0
        if isinstance(data, dict) and ('image' in data or 'images' in data)and images_left.shape[0] > 0 and len(image_list) > 0:
            images_right = images_left + self.multimodal_cfg['image_token_len'] + 1
            num_images_right = torch.where(images_right < data_dict["input_ids"].shape[0])[0].shape[0]
            if num_images_right < images_left.shape[0]:
                data_dict["input_ids"] = torch.cat([data_dict["input_ids"][:images_left[num_images_right]], torch.tensor([self.tokenizer.eos_token_id])])
                data_dict["labels"] = torch.cat([data_dict["labels"][:images_left[num_images_right]], torch.tensor([self.tokenizer.eos_token_id])])
        
        # ensure it's vl data firstly, then ensure there are images secondly
        if isinstance(data, dict) and 'image' in data and num_images_right > 0 and len(image_list) > 0:
            data_dict['image'] = image_list[:num_images_right]
        else:
            data_dict['image'] = [torch.zeros(3, self.image_size, self.image_size)]
        return data_dict

