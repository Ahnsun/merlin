import io
import json
import torch
import transformers
import random
import boto3
import smart_open
import megfile
import time

from megfile import s3_path
from loguru import logger
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from mmgpt.utils.constants import *
from mmgpt.utils.mm_utils import expand2square

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    def __init__(
        self, 
        datasets: str,
        tokenizer: transformers.PreTrainedTokenizer,
        multimodal_cfg: dict
    ):
        super(BaseDataset, self).__init__()
        self.tokenizer = tokenizer
        self.multimodal_cfg = multimodal_cfg

        self.session = boto3.Session(aws_access_key_id=s3_path.get_access_token()[0],
                                     aws_secret_access_key=s3_path.get_access_token()[1])
        self.endpoint_url = s3_path.get_endpoint_url()
        
        self.image_size = self.multimodal_cfg['image_size']
        self.processor = self.multimodal_cfg['image_processor']

        self.im_patch_token, self.im_start_token, self.im_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    
    def load_json(self, file_path):
        data = []
        with megfile.smart_open(file_path, "r") as f:
        # with smart_open.open(file_path, 'rb', transport_params={'client': self.session.client('s3', endpoint_url=self.endpoint_url)}) as f:
            try:
                lines = json.load(f)
                if isinstance(lines, list):
                    data.extend(lines)
                else:
                    data.append(lines)
            except:
                pass
        return data

    def parallel_load_json(self, json_paths, max_workers=8):
        list_data_dict = []
        results = Parallel(n_jobs=max_workers)(
            delayed(self.load_json)(json_path) for json_path in json_paths
        )
        for _, _r in enumerate(results):
            list_data_dict.extend(_r)
        return list_data_dict

    def load_images(self, image_path):
        if 's3://' in image_path:
            # with megfile.smart_open(image_path, "rb") as f:
            with smart_open.open(image_path, 'rb', transport_params={'client': self.session.client('s3', endpoint_url=self.endpoint_url)}) as f:
                bytes_data = f.read()
            image = Image.open(io.BytesIO(bytes_data), "r").convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
            
        return image
        
    def box_shuffle_and_sample(self, boxes, conversations):
        # deprecated, recommand to pre-process instead of online processing
        if self.multimodal_cfg['box_limit'] > 0:
            boxes = boxes[:self.multimodal_cfg['box_limit']]
            conversations = conversations[:self.multimodal_cfg['box_limit'] * 2]
            
        elif self.multimodal_cfg['box_limit'] < 0:
            rand_idx = list(range(len(boxes)))
            random.shuffle(rand_idx)
            box_limit = random.randint(1, abs(self.multimodal_cfg['box_limit']))
            sampled_idx = rand_idx[:box_limit]

            new_boxes, new_conversations = [], []
            for idx in sampled_idx:
                new_boxes.append(boxes[idx])
                new_conversations.extend(conversations[idx * 2: idx * 2 + 2])

            boxes = new_boxes
            conversations = new_conversations

        if DEFAULT_IMAGE_TOKEN not in conversations[0]['value']:
            conversations[0]['value'] = f"Given an image\n{DEFAULT_IMAGE_TOKEN}. " + conversations[0]['value']

        return boxes, conversations

    def merge_conversations(self, data, merge_round):
        if merge_round == 0:
            return data
        elif merge_round > 0:
            merged_data = []
            chunks = [data[x:x+merge_round] for x in range(0, len(data), merge_round)]
            for chunk in chunks:
                image_list = []
                conversation_list = []
                for line in chunk:
                    image_list.append(line['image'])
                    conversation_list.extend(line['conversations'])
                merged_data.append({
                    'images': image_list,
                    'conversations': conversation_list
                })
        else:
            merged_data = []
            idx = 0
            while idx < len(data):
                rand_round = random.randint(1, 5)
                image_list = []
                conversation_list = []
                for line in data[idx:idx+rand_round]:
                    image_list.append(line['image'])
                    conversation_list.extend(line['conversations'])
                merged_data.append({
                    'images': image_list,
                    'conversations': conversation_list
                })
                idx += rand_round

            import torch.distributed as dist
            min_len_merged_data = torch.tensor(len(merged_data)).cuda()
            dist.all_reduce(min_len_merged_data, op=dist.ReduceOp.MIN)
            merged_data = merged_data[:min_len_merged_data.item()]

        logger.info(f'after merging per {merge_round} conversations, data are shrinked from {len(data)} to {len(merged_data)}.')
        return merged_data

    def box_processor(self, boxes_list, image_wh_list, image_path):
        assert len(boxes_list) == len(image_wh_list)
        assert self.multimodal_cfg['image_aspect_ratio'] in ['pad', 'resize']

        box_text_list = []
        for boxes, (im_w, im_h) in zip(boxes_list, image_wh_list):
            boxes = torch.tensor(boxes, dtype=torch.float32)

            if 'OpenImages' in image_path:
                boxes[:, 0::2] *= im_w
                boxes[:, 1::2] *= im_h
                boxes[:, 0::2].clamp_(min=0, max=im_w)
                boxes[:, 1::2].clamp_(min=0, max=im_h)
            else:
                boxes[:, 2:] += boxes[:, :2]
                boxes[:, 0::2].clamp_(min=0, max=im_w)
                boxes[:, 1::2].clamp_(min=0, max=im_h)
            
            if self.multimodal_cfg['image_aspect_ratio'] == 'pad':
                max_hw = max(im_w, im_h)
                boxes /= torch.tensor([max_hw, max_hw, max_hw, max_hw], dtype=torch.float32)
            elif self.multimodal_cfg['image_aspect_ratio'] == 'resize':
                boxes /= torch.tensor([im_w, im_h, im_w, im_h], dtype=torch.float32)
            else:
                raise ValueError("Unsupported type.")

            # box_text_list.extend([f"[{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]" for box in boxes.numpy()])
            box_text_list.extend(
                [
                    f"[{int(box[0]*1000):03d}, {int(box[1]*1000):03d}, {int(box[2]*1000):03d}, {int(box[3]*1000):03d}]" 
                    for box in boxes.numpy()
                ]
            )

        return box_text_list

    def image_processor(self, image):
        if self.multimodal_cfg['image_aspect_ratio'] == 'keep':
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            max_len, min_len = self.image_size * 2, self.image_size
            shortest_edge = int(min(max_len / aspect_ratio, min_len))
            image = self.processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]

        elif self.multimodal_cfg['image_aspect_ratio'] == 'pad':
            image = expand2square(image, tuple(int(x*255) for x in self.processor.image_mean))
            image = self.processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": self.image_size})['pixel_values'][0]

        elif self.multimodal_cfg['image_aspect_ratio'] == 'resize':
            image = image.resize((self.image_size, self.image_size))
            image = self.processor.preprocess(image, return_tensors='pt', do_resize=False, do_center_crop=False)['pixel_values'][0]
        else:
            image = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # assert (image.shape[1]//14) * (image.shape[2]//14) == self.multimodal_cfg['image_token_len']
        return image

