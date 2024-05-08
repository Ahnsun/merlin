from mmgpt.model.projector.mlp_projector import MLPProjector
from mmgpt.model.projector.conv_projector import ConvProjector
from mmgpt.model.projector.sam_projector import SAMProjector
from mmgpt.model.projector.qwen_projector import QWenProjector
from mmgpt.model.projector.qwen_sampler_projector import QWenSamplerProjector


def build_projector(projector_cfg, vision_hidden_size, lm_hidden_size):
    projector = getattr(projector_cfg, 'projector', None)
    if projector == "mlp":
        return MLPProjector(
            args=projector_cfg, 
            vision_hidden_size=vision_hidden_size, 
            lm_hidden_size=lm_hidden_size
        )
    elif projector == "conv":
        return ConvProjector(
            args=projector_cfg, 
            vision_hidden_size=vision_hidden_size, 
            lm_hidden_size=lm_hidden_size,
            conv_stride=projector_cfg.conv_stride
        )
    elif projector == "sam":
        return SAMProjector(
            args=projector_cfg, 
            vision_hidden_size=vision_hidden_size, 
            lm_hidden_size=lm_hidden_size
        )
    elif projector == "qwen":
        return QWenProjector(
            args=projector_cfg, 
            vision_hidden_size=vision_hidden_size, 
            lm_hidden_size=lm_hidden_size
        )
    elif projector == "qwen-sampler":
        return QWenSamplerProjector(
            args=projector_cfg, 
            vision_hidden_size=vision_hidden_size, 
            lm_hidden_size=lm_hidden_size
        )

    raise ValueError(f'Unknown projector: {projector}')