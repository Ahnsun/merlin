import os
import argparse
import torch
import json
import math
import base64
import pandas as pd

from io import BytesIO
from tqdm import tqdm
from PIL import Image

from mmgpt.model.builder import build_model_tokenizer
from mmgpt.utils.arguments import *
from mmgpt.utils.constants import DEFAULT_IM_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_EOS_TOKEN
from mmgpt.utils.conversation import conv as default_conversation
from mmgpt.utils.utils import disable_torch_init
from mmgpt.utils.mm_utils import KeywordsStoppingCriteria, load_image, expand2square, process_image
from mmgpt.utils.evaluation_tools.mmbench_evaluator import eval_result


all_options = ['A', 'B', 'C', 'D']


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def prepare_input(tokenizer, data_args, line, options):
    question = line['question']
    hint = line['hint']

    if not is_none(hint):
        question = hint + '\n' + question
    for option_char, option in zip(all_options[:len(options)], options):
        question = question + '\n' + option_char + '. ' + option

    if data_args.use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IM_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        qs = DEFAULT_IM_PATCH_TOKEN * 256 + '\n' + question

    if 'cn' in data_args.eval_file_path.lower():
        qs = qs + '\n' + "请直接回答选项字母。"
    else:
        qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()

    image = load_image_from_base64(line['image'])
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
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, tokenizer, data_args = build_model_tokenizer(model_args, data_args, training_args)

    # read questions for infering
    df = pd.read_table(os.path.expanduser(data_args.eval_file_path))
    questions = get_chunk(df, 1, 0)

    # for saving and evaluation
    results = df.copy()
    results = results.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
    results.insert(6, 'prediction', None)

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        input_ids, image_tensor, stopping_criteria = prepare_input(
            tokenizer, data_args, row, get_options(row, all_options)
        )

        with torch.inference_mode():
            if data_args.use_beam_search:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    num_beams=5,
                    temperature=0.2,
                    # repetition_penalty=1.0,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
            else:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    # repetition_penalty=1.0,
                    max_new_tokens=1024,
                    # use_cache=True,
                    stopping_criteria=[stopping_criteria])
            
            outputs = decode(tokenizer, input_ids, output_ids)
            # print(f"{row['index']}: Question: {row['question']} Ground Truth: {row['answer']} Answer: {outputs}")
            results.loc[results['index'] == row['index'], 'prediction'] = outputs

    if 'cn' in data_args.eval_file_path.lower():
        result_filename = "mmbench_cn.xlsx"
    else:
        result_filename = "mmbench.xlsx"

    if 'test' in data_args.eval_file_path.lower():
        result_filename = 'test_' + result_filename

    results.to_excel(os.path.join(model_args.model_name_or_path, result_filename), index=False, engine='openpyxl')

    # simulated evaluation
    acc, l2, leaf = eval_result(
        pred_file=os.path.join(model_args.model_name_or_path, result_filename), 
        gt_file=data_args.eval_file_path, 
        tmp_dir=model_args.model_name_or_path
    )


if __name__ == "__main__":
    eval()
