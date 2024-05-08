from mmgpt.model.vision_encoder.clip_encoder import CLIPVisionTower
from mmgpt.model.vision_encoder.sam_encoder import SAMVisionTower
from mmgpt.model.vision_encoder.qwen_encoder import QWenVisionTower
from mmgpt.model.vision_encoder.qwen_nosampler_encoder import QWenNoSamplerVisionTower
# from mmgpt.model.vision_encoder.qwen_nosampler_encoder_flash_attn import QWenNoSamplerVisionTower

def build_vision_tower(vision_tower_cfg):
    vision_tower = getattr(vision_tower_cfg, 'vision_tower', None)
    
    if "qwen" in vision_tower.lower():
        # return QWenVisionTower(args=vision_tower_cfg)
        return QWenNoSamplerVisionTower(args=vision_tower_cfg)
    elif "sam" in vision_tower.lower():
        return SAMVisionTower(args=vision_tower_cfg)
    elif "clip" in vision_tower.lower():
        return CLIPVisionTower(args=vision_tower_cfg)

    raise ValueError(f'Unknown vision tower: {vision_tower}')