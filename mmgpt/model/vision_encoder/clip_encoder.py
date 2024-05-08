import os
import json
import torch
import torch.nn as nn

from loguru import logger
from collections import defaultdict
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.vision_tower_name = args.vision_tower
        self.select_layer = args.vision_select_layer
        self.select_feature = args.vision_select_feature
        self.freeze_vision_tower = args.freeze_vision_tower
        self.conv_stride = args.conv_stride

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        self.load_weights(args.model_name_or_path)
    
    def load_weights(self, model_path):
        if os.path.exists(os.path.join(model_path, 'pytorch_model.bin.index.json')):
            model_indices = json.load(open(os.path.join(model_path, 'pytorch_model.bin.index.json')))

            ckpt_to_key = defaultdict(list)
            for k, v in model_indices['weight_map'].items():
                if 'vision_tower' in k:
                    ckpt_to_key[v].append(k)

            if len(ckpt_to_key) == 0:
                return

            vision_tower_weight = {}
            prefix_length = len('model.vision_tower.')

            for ckpt_name, weight_keys in ckpt_to_key.items():
                ckpt = torch.load(os.path.join(model_path, ckpt_name), map_location='cpu')
                for k in weight_keys:
                    vision_tower_weight[k[prefix_length:]] = ckpt[k]

                self.load_state_dict(vision_tower_weight, strict=True)
                logger.info(f"vision_tower weights (from {os.path.join(model_path, ckpt_name)}) are loaded!")
        
        elif os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
            vision_tower_weight = {} 
            prefix_length = len('model.vision_tower.')

            ckpt = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')
            for k in ckpt.keys():
                if 'vision_tower' in k:
                    vision_tower_weight[k[prefix_length:]] = ckpt[k]

            if len(vision_tower_weight) == 0:
                return

            self.load_state_dict(vision_tower_weight, strict=True)
            logger.info(f"vision_tower weights (from {os.path.join(model_path, 'pytorch_model.bin')}) are loaded!")

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select fe.       ature: {self.select_feature}')
        return image_features

    def forward(self, images):
        with torch.set_grad_enabled(not self.freeze_vision_tower):
            concat_images = torch.cat([image for image in images], dim=0).to(device=self.device, dtype=self.dtype) 
            split_sizes = [image.shape[0] for image in images]

            images_forward_out = self.vision_tower(concat_images, output_hidden_states=True)
            image_features = self.feature_select(images_forward_out).to(images[0].dtype)
            image_features = torch.split(image_features, split_sizes, dim=0)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
        # return [torch.zeros(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)] # for resampler

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size // self.conv_stride) ** 2
