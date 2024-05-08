# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import pathlib
import torch
import transformers

from mmgpt.data.builder import build_dataloader
from mmgpt.engine.train.trainer import MMGPTTrainer
from mmgpt.model.builder import build_model_tokenizer
from mmgpt.utils.arguments import *
from mmgpt.utils.constants import *
from mmgpt.utils.peft_utils import *
from mmgpt.utils.logger import setup_logger, log_model_parameters
from mmgpt.utils.dist_utils import get_rank


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    setup_logger(training_args.output_dir, get_rank())

    model, tokenizer, data_args = build_model_tokenizer(model_args, data_args, training_args)
    
    log_model_parameters(model)

    model.config.use_cache = False

    dataloader = build_dataloader(
        tokenizer=tokenizer,
        data_args=data_args
    )
    
    trainer = MMGPTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **dataloader
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        # trainer._save(output_dir=training_args.output_dir)
        trainer._safe_save(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
