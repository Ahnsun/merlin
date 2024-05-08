from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/data/hypertext/data/data/llm/cnft/llama_7b_cnft")
    vision_tower: Optional[str] = field(default="/data/hypertext/data/data/llm/clip/vit-large-patch14")
    projector: Optional[str] = field(default="mlp")
    freeze_vision_tower: bool = field(default=False)
    freeze_projector: bool = field(default=False)
    freeze_lm_model: bool = field(default=False)
    vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    vision_select_feature: Optional[str] = field(default="patch")
    use_im_start_end: bool = field(default=True)
    conv_stride: int = 1


@dataclass
class DataArguments:
    datasets: str = field(default=None, metadata={"help": "combinations of the training data."})
    conversation_datasets: str = field(default=None)
    pair_webdatasets: str = field(default=None)
    pair_token_webdatasets: str = field(default=None)
    interpair_webdatasets: str = field(default=None)
    interleave_webdatasets: str = field(default=None)
    image_size: int = 224
    num_patches: int = 256
    image_aspect_ratio: str = 'square'
    box_limit: int = 0
    eval_image_path: Optional[str] = field(default="/data/public/lucaszhao/workspace/MMGPT/mmgpt/serve/examples/ocr2.png")
    eval_file_path: Optional[str] = field(default="/data/hypertext/data/data/dataset/MM-VET/MMGPT_mm-vet.json")
    eval_question: Optional[str] = field(default="what can you see from this image?")
    use_beam_search: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    llrd: bool = False
    llm_llrd: bool = False
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"