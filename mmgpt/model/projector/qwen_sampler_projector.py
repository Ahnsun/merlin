import os
import math
import json
import torch
import torch.nn as nn

from loguru import logger
from functools import partial
from collections import defaultdict

from mmgpt.model.vision_encoder.qwen_nosampler_encoder import Resampler


class QWenSamplerProjector(nn.Module):
    def __init__(self, args, vision_hidden_size, lm_hidden_size):
        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.freeze_projector = args.freeze_projector

        self.attn_pool = Resampler(
            grid_size=int(math.sqrt(256)),
            embed_dim=vision_hidden_size,
            num_heads=vision_hidden_size // 128,
            kv_dim=1024,
            norm_layer=norm_layer,
        )
        self.ln_post = norm_layer(vision_hidden_size)
        self.proj = nn.Parameter((vision_hidden_size** -0.5) * torch.randn(vision_hidden_size, lm_hidden_size))

        self.load_weights(args.vision_tower, args.model_name_or_path)
    
    def load_weights(self, vit_path, model_path):
        # # init with vision tower weight by default
        # if os.path.exists(os.path.join(vit_path, 'pytorch_model.bin')):
        #     ckpt = torch.load(os.path.join(vit_path, 'pytorch_model.bin'), map_location='cpu')

        #     pretrained_weights = {}
        #     for layer_name in ['attn_pool', 'ln_post', 'proj']:
        #         prefix = f'transformer.visual.'
        #         for k in ckpt.keys():
        #             if prefix + layer_name in k:
        #                 pretrained_weights[k[len(prefix):]] = ckpt[k]

        #     if len(pretrained_weights) == 0:
        #         return
            
        #     self.load_state_dict(pretrained_weights, strict=True)
        #     logger.info(f"projector (with resampler) weights (from {os.path.join(vit_path, 'pytorch_model.bin')}) have been loaded!")

        # init with vision tower weight contained by mllm
        if os.path.exists(os.path.join(model_path, 'pytorch_model.bin.index.json')):
            model_indices = json.load(open(os.path.join(model_path, 'pytorch_model.bin.index.json')))

            ckpt_to_key = defaultdict(list)
            for k, v in model_indices['weight_map'].items():
                if 'projector' in k:
                    ckpt_to_key[v].append(k)

            if len(ckpt_to_key) == 0:
                return

            projector_weight = {}
            prefix_length = len('model.projector.')

            for ckpt_name, weight_keys in ckpt_to_key.items():
                ckpt = torch.load(os.path.join(model_path, ckpt_name), map_location='cpu')
                for k in weight_keys:
                    projector_weight[k[prefix_length:]] = ckpt[k]

                self.load_state_dict(projector_weight, strict=True)
                logger.info(f"projector weights (with resampler) (from {os.path.join(model_path, ckpt_name)}) are loaded!")
        
        elif os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
            projector_weight = {}
            prefix_length = len('model.projector.')

            ckpt = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')
            for k in ckpt.keys():
                if 'projector' in k:
                    projector_weight[k[prefix_length:]] = ckpt[k]

            if len(projector_weight) == 0:
                return

            self.load_state_dict(projector_weight, strict=True)
            logger.info(f"projector (with resampler) weights (from {os.path.join(model_path, 'pytorch_model.bin')}) are loaded!")

    def forward(self, features):
        with torch.set_grad_enabled(not self.freeze_projector):
            projected_features = []
            for feature in features:
                feature = self.attn_pool(feature)
                feature = self.ln_post(feature)
                feature = feature @ self.proj
                projected_features.append(feature)
            return projected_features
    
    @property
    def dtype(self):
        return self.projector.dtype

    @property
    def device(self):
        return self.projector.device