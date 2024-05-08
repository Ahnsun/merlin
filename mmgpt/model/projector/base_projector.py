import os
import json
import torch
import torch.nn as nn

from loguru import logger
from collections import defaultdict


class BaseProjector(nn.Module):
    
    def load_weights(self, model_path):
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
                logger.info(f"projector weights (from {os.path.join(model_path, ckpt_name)}) are loaded!")
        
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
            logger.info(f"projector weights (from {os.path.join(model_path, 'pytorch_model.bin')}) are loaded!")

    def forward(self, features):
        pass
    
    @property
    def dtype(self):
        return self.projector.dtype

    @property
    def device(self):
        return self.projector.device