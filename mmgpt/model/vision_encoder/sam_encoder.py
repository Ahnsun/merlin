import torch
import torch.nn as nn

from transformers import SamModel, SamImageProcessor
# from mmgpt.model.vision_encoder.blip_process import BlipImageEvalProcessor
# from mmgpt.model.vision_encoder.blip_process import BlipImageTrainProcessor
from mmgpt.model.vision_encoder.utils.image_encoder import build_sam_vit_b


class SAMVisionTower(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.vision_tower_name = args.vision_tower
        self.select_layer = args.vision_select_layer
        self.select_feature = args.vision_select_feature
        self.freeze_vision_tower = args.freeze_vision_tower

        self.image_processor = SamImageProcessor.from_pretrained(self.vision_tower_name)
        # self.vision_tower = SamModel.from_pretrained(self.vision_tower_name).vision_encoder
        # self.name = 'sam'
        self.vision_tower = build_sam_vit_b(checkpoint= '/data/public/ucaswei/codes/LAVIS-main/lavis/models/blip2_models/backbone/sam_vit_b_01ec64.pth')

    def feature_select(self, image_forward_outs):
        # image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_forward_outs.last_hidden_state
        return image_features

    def forward(self, images):
        with torch.set_grad_enabled(not self.freeze_vision_tower):
            image_features = []
            for image in images:
                # image_forward_out = self.vision_tower(image, output_hidden_states=True)
                # image_feature = self.feature_select(image_forward_out)
                # image_features.append(image_feature)
                # image_feature = self.vision_tower(image).flatten(2).permute(0, 2, 1)
                image_feature = self.vision_tower(image)
                # print(image_feature.shape)
                # exit()
                image_features.append(image_feature)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

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
        return 256
