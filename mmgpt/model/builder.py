# ------------------------------------------------------------------------------------------------
# Copyright (c) 2023 Megvii, Inc. All rights reserved.
# ------------------------------------------------------------------------------------------------
import os
import math
import torch

from loguru import logger
from transformers import AutoConfig, AutoTokenizer

from mmgpt.model.mmgpt.llama_mmgpt import MMGPTLlamaForCausalLM
from mmgpt.model.mmgpt.opt_mmgpt import MMGPTOPTForCausalLM
from mmgpt.utils.peft_utils import *
from mmgpt.utils.constants import *
import ipdb


def build_model_tokenizer(model_args, data_args, training_args):
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )

    if 'baichuan' in model_args.model_name_or_path:
        if '2-7b' in model_args.model_name_or_path:
            from mmgpt.model.mmgpt.baichuan2_7b_mmgpt import MMGPTBaiChuanForCausalLM
        elif '2-13b' in model_args.model_name_or_path:
            from mmgpt.model.mmgpt.baichuan2_13b_mmgpt import MMGPTBaiChuanForCausalLM
        elif '7b' in model_args.model_name_or_path:
            from mmgpt.model.mmgpt.baichuan7b_mmgpt import MMGPTBaiChuanForCausalLM
        elif '13b' in model_args.model_name_or_path:
            from mmgpt.model.mmgpt.baichuan13b_mmgpt import MMGPTBaiChuanForCausalLM
        else:
            raise ValueError('Not Supported Baichuan version')
        
        # Set RoPE scaling factor
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            logger.info(f'RoPE has been scaled from {orig_ctx_len} to {training_args.model_max_length}')
            
        model = MMGPTBaiChuanForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=True,
            cache_dir=training_args.cache_dir,
        )
        # Tie the weights
        # model.tie_weights()
    elif 'phi' in model_args.model_name_or_path:
        # from mmgpt.model.mmgpt.phi2_mmgpt import MMGPTPhiForCausalLM
        from mmgpt.model.mmgpt.phi2_mmgpt_v1 import MMGPTPhiForCausalLM
        model = MMGPTPhiForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=True,
            cache_dir=training_args.cache_dir,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
        )
    elif 'opt' in model_args.model_name_or_path:
        model = MMGPTOPTForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
        )
    else:
        model = MMGPTLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        # same with llava-phi
        use_fast=True if "phi" in model_args.model_name_or_path else False,
    )

    # deal with llama model that special tokens are not claimed
    if "models--decapoda-research--llama-7b-hf" in model_args.model_name_or_path:
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })
        
    # NOTE: if the token_id exceed the vocab_size will cause failing in training process! we need add special config and resize the embedding size!
    tokenizer.pad_token = tokenizer.unk_token
    # tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    logger.info(tokenizer)

    # add a hook for the training of token embedding in isolation
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    dtype = torch.float32
    if training_args.bf16:
        dtype = torch.bfloat16
    if training_args.fp16:
        dtype = torch.float16

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model.to(dtype)

        logger.info("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if model_args.freeze_lm_model:
        model.requires_grad_(False)
    
    if model_args.vision_tower is not None:
        model.build_vision_tokenizer(
            model_args=model_args, 
            data_args=data_args,
            training_args=training_args,
            tokenizer=tokenizer,
        )

        # when unfreezing llm, all parameters must be flattened by FSDP
        if model_args.freeze_lm_model:
            # set gradient for vision tower (last layer is always detached)
            model.get_model().vision_tower.requires_grad_(not model_args.freeze_vision_tower)
            if hasattr(model.get_model().vision_tower.vision_tower, 'vision_model'):
                model.get_model().vision_tower.vision_tower.vision_model.encoder.layers[-1].requires_grad_(False)
                model.get_model().vision_tower.vision_tower.vision_model.post_layernorm.requires_grad_(False)
            elif hasattr(model.get_model().vision_tower.vision_tower, 'neck'):
                model.get_model().vision_tower.vision_tower.neck.requires_grad_(False)

            # set gradient for projector
            model.get_model().projector.requires_grad_(not model_args.freeze_projector)

            # set gradient for token embedding
            if model_args.use_im_start_end:
                for p in model.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in model.get_output_embeddings().parameters():
                    p.requires_grad = False

    logger.info(model)
    model.to(dtype=dtype, device=training_args.device)

    return model, tokenizer, data_args
