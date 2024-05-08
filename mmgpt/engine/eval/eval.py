import os
import argparse
import torch
import json

from tqdm import tqdm

from mmgpt.model.builder import build_model_tokenizer
from mmgpt.utils.arguments import *
from mmgpt.utils.constants import DEFAULT_IM_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_EOS_TOKEN
from mmgpt.utils.conversation import conv as default_conversation
from mmgpt.utils.utils import disable_torch_init
from mmgpt.utils.mm_utils import KeywordsStoppingCriteria, load_image, expand2square


def process_image(image, processor, image_size=448, mode='resize'):
    if mode == 'keep':
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = image_size * 2, image_size
        shortest_edge = int(min(max_len / aspect_ratio, min_len))
        image = processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values']

    elif mode == 'pad':
        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": image_size})['pixel_values']

    elif mode == 'resize':
        image = image.resize((image_size, image_size))
        image = processor.preprocess(image, return_tensors='pt', do_resize=False, do_center_crop=False)['pixel_values']
    else:
        image = processor.preprocess(image, return_tensors='pt')['pixel_values']

    return image


def prepare_input(tokenizer, data_args, eval_question, eval_image):
    if data_args.use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IM_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + '\n' + eval_question
    else:
        qs = DEFAULT_IM_PATCH_TOKEN * 256 + '\n' + eval_question

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()

    image = load_image(eval_image)
    image_tensor = process_image(
        image, 
        data_args.image_processor,
        data_args.image_size,
        data_args.image_aspect_ratio,
    ).to(torch.bfloat16).cuda()
    # image_tensor = data_args.image_processor.preprocess(image.resize((data_args.image_size, data_args.image_size)), return_tensors='pt')['pixel_values'].to(torch.bfloat16).cuda()

    stop_str = DEFAULT_EOS_TOKEN
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    return input_ids, [image_tensor], stopping_criteria


def decode(tokenizer, input_ids, output_ids):
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    stop_str = DEFAULT_EOS_TOKEN
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def eval():
    disable_torch_init()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model, tokenizer, data_args = build_model_tokenizer(*parser.parse_args_into_dataclasses())

    input_ids, image_tensor, stopping_criteria = prepare_input(
        tokenizer, 
        data_args,
        data_args.eval_question,
        data_args.eval_image_path
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=1.0,
            repetition_penalty=1.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = decode(tokenizer, input_ids, output_ids)
    print(outputs)


if __name__ == "__main__":
    eval()
