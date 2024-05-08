import torch
import torch.nn as nn

from loguru import logger
from abc import ABC, abstractmethod

from mmgpt.model.projector.builder import build_projector
from mmgpt.model.vision_encoder.builder import build_vision_tower
from mmgpt.utils.constants import *


class MMGPTMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def encode_images(self, images):
        image_features = self.get_model().vision_tower(images)
        image_features = self.get_model().projector(image_features)
        return image_features

    def build_vision_tokenizer(self, model_args, data_args, training_args, tokenizer):
        # build vision tower and projector
        vision_tower = build_vision_tower(model_args)

        if hasattr(vision_tower, 'hidden_size'):
            vision_hidden_size = vision_tower.hidden_size
        elif hasattr(vision_tower, 'config.hidden_size'):
            vision_hidden_size = vision_tower.config.hidden_size
        elif hasattr(vision_tower, 'output_dim'):
            vision_hidden_size = vision_tower.output_dim
        else:
            vision_hidden_size = 256
            # raise ValueError('Not Defined Vision Hidden Size! Please hand-set the value in this line.')
        
        projector = build_projector(
            model_args, 
            vision_hidden_size, 
            self.config.hidden_size
        )

        self.get_model().vision_tower = vision_tower
        self.get_model().projector = projector

        # update data arguments with vision tower configs
        data_args.image_token_len = vision_tower.num_patches
        data_args.image_processor = vision_tower.image_processor
        data_args.use_im_start_end = model_args.use_im_start_end
        self.use_im_start_end = model_args.use_im_start_end
        self.use_beam_search = data_args.use_beam_search
        logger.info(vision_tower.image_processor)

        # add image patch token <im_patch> (optional, to be removed)
        tokenizer.add_tokens([DEFAULT_IM_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        self.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_PATCH_TOKEN])[0]

        # add image start token <im_start> and end token <im_end>
        if self.use_im_start_end:
            self.num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            self.im_start_token, self.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
            logger.info(f'{self.num_new_tokens} new tokens are added to be trained.')
            
            # it mean some new special tokens are added and are supposed to be learned
            if self.num_new_tokens > 0:
                # these new added embeddings are initalized with average weights
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-self.num_new_tokens:] = input_embeddings_avg
                output_embeddings[-self.num_new_tokens:] = output_embeddings_avg

                if model_args.freeze_lm_model:
                    self.get_model().orig_embeds_params = self.get_input_embeddings().weight.data.clone().to(device=training_args.device)


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = getattr(self.get_model(), 'vision_tower', None)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
            
        image_features = self.encode_images(images)

        # HACK: replace back original embeddings for pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        if orig_embeds_params is not None:
            with torch.no_grad():
                self.get_input_embeddings().weight[:-self.num_new_tokens] = orig_embeds_params[:-self.num_new_tokens].data

        if hasattr(self.get_model(), 'decoder'):
            inputs_embeds = self.get_model().decoder.embed_tokens(input_ids)
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        new_input_embeds = []
        for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_features):
            # index = torch.where(cur_input_ids == self.im_patch_token)[0]
            # cur_input_embeds.scatter_(1, index, cur_image_features)

            if (cur_input_ids == self.im_patch_token).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = cur_input_embeds + (0. * self.get_model().projector(vision_tower.dummy_feature)[0]).sum()
                new_input_embeds.append(cur_input_embeds)
                continue

            if self.use_im_start_end:
                if (cur_input_ids == self.im_start_token).sum() != (cur_input_ids == self.im_end_token).sum():
                    raise ValueError(f"The number of image start tokens ({(cur_input_ids == self.im_start_token).sum()}) \
                                     and image end tokens ({(cur_input_ids == self.im_end_token).sum()}) should be the same.")
                
                image_start_tokens = torch.where(cur_input_ids == self.im_start_token)[0]
                for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_image_features):
                    per_cur_image_features = per_cur_image_features.to(device=cur_input_embeds.device)
                    num_patches = per_cur_image_features.shape[0]

                    if cur_input_ids[image_start_token_pos + num_patches + 1] != self.im_end_token:
                        raise ValueError("The image end token should follow the image start token.")
                    
                    cur_input_embeds = torch.cat(
                        (
                            cur_input_embeds[:image_start_token_pos+1], 
                            per_cur_image_features, 
                            cur_input_embeds[image_start_token_pos + num_patches + 1:]
                        ), 
                        dim=0
                    )
            else:
                raise NotImplementedError
                # num_patches = cur_image_features.shape[0]

                # if (cur_input_ids == self.im_patch_token).sum() != num_patches:
                #     raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                
                # masked_indices = torch.where(cur_input_ids == self.im_patch_token)[0]
                # mask_index_start = masked_indices[0]

                # if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                #     raise ValueError("The image patch tokens should be consecutive.")
                
                # cur_input_embeds = torch.cat(
                #     (
                #         cur_input_embeds[:mask_index_start], 
                #         cur_image_features, 
                #         cur_input_embeds[mask_index_start+num_patches:]
                #     ), 
                #     dim=0
                # )

            new_input_embeds.append(cur_input_embeds)

        inputs_embeds = torch.stack(new_input_embeds, dim=0)

        if self.use_beam_search:
            inputs_embeds = inputs_embeds.repeat_interleave(5, dim=0)

        return None, attention_mask, past_key_values, inputs_embeds, labels
