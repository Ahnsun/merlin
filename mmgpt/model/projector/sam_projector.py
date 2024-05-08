import math
import torch
import torch.nn as nn


class SAMProjector(nn.Module):
    def __init__(self, args, vision_hidden_size, lm_hidden_size):
        super().__init__()

        self.freeze_projector = args.freeze_projector

        self.projector = nn.Sequential(
            nn.Conv2d(
                in_channels=256, 
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),            
            nn.Conv2d(
                in_channels=512, 
                out_channels=1024,
                kernel_size=3,
                stride=2,
                padding=1, 
                bias=False
            ),
        )
        self.mlp = nn.Linear(1024, 4096)

    def forward(self, features):
        with torch.set_grad_enabled(not self.freeze_projector):
            projected_features = []
            for feature in features:
                B, C, H, W = feature.shape
                projected_feature = self.projector(feature)
                projected_feature = projected_feature.view(B, -1, H * W // 4 // 4).permute(0, 2, 1)
                projected_feature = self.mlp(projected_feature)
                projected_features.append(projected_feature)
            return projected_features
    
    @property
    def dtype(self):
        return self.projector.dtype

    @property
    def device(self):
        return self.projector.device