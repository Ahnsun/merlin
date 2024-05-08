import os
import json
import torch
import torch.nn as nn

from loguru import logger
from collections import defaultdict
from mmgpt.model.projector.base_projector import BaseProjector


class MLPProjector(BaseProjector):
    def __init__(self, args, vision_hidden_size, lm_hidden_size):
        super().__init__()

        self.freeze_projector = args.freeze_projector
        self.projector = nn.Linear(vision_hidden_size, lm_hidden_size)
        self.load_weights(args.model_name_or_path)

    def forward(self, features):
        with torch.set_grad_enabled(not self.freeze_projector):
            return [
                self.projector(feature) for feature in features
            ]