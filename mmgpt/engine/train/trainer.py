import os
import torch

from loguru import logger
from transformers import Trainer
from transformers.trainer import has_length

from mmgpt.data.sampler import LengthGroupedSampler
from mmgpt.utils.llrd_utils import *
from mmgpt.utils.peft_utils import *


class MMGPTTrainer(Trainer):
    def _get_train_sampler(self):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()
        
    def _safe_save(self, output_dir: str):
        """Collects the state dict and dump to disk."""
        if self.deepspeed:
            torch.cuda.synchronize()
            self.save_model(output_dir)
            return
    
        state_dict = self.model.state_dict()
        if self.args.should_save:
            cpu_state_dict = {
                key: value.cpu()
                for key, value in state_dict.items()
            }
            del state_dict
            self._save(output_dir, state_dict=cpu_state_dict)  # noqa

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            if self.args.llrd:
                lr_scale_func = vit_lr_scale_func
            elif self.args.llm_llrd:
                lr_scale_func = llm_lr_scale_func
            else:
                lr_scale_func = None

            optimizer_grouped_parameters = get_param_groups(
                opt_model, 
                None, 
                lr_scale_func, 
                self.args.learning_rate, 
                self.args.weight_decay
            )
            logger.info(f"->> Number of Optimizer Groups: {len(optimizer_grouped_parameters)}")
            for idx, group in enumerate(optimizer_grouped_parameters):
                logger.info(f"*********>> {idx}: {len(group['params'])} groups of parameters maintains a learning rate of {group['lr']}")
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer