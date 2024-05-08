import os
import argparse
import torch
import json

from tqdm import tqdm
import os.path as osp
from torchvision.transforms import ToPILImage, PILToTensor
from torchvision.utils import draw_bounding_boxes as _draw_bounding_boxes
from torchvision.ops import box_convert
import numpy as np
from PIL import Image, ImageFont
from io import BytesIO
import requests
import re

from mmgpt.model.builder import build_model_tokenizer
from mmgpt.utils.arguments import *
from mmgpt.utils.constants import DEFAULT_IM_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_EOS_TOKEN
from mmgpt.utils.conversation import conv as default_conversation
from mmgpt.utils.utils import disable_torch_init
from mmgpt.utils.mm_utils import KeywordsStoppingCriteria, load_image, expand2square

def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h

    if x1 > x2 or y1 > y2:
        x2 = x1 + x2
        y2 = y1 + y2
    
    box = x1, y1, x2, y2

    return box


def draw_bounding_boxes(
        image,
        boxes,
        **kwargs,
):
    if isinstance(image, Image.Image):
        image = PILToTensor()(image)
    assert isinstance(image, torch.Tensor), ""

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes)
    assert isinstance(boxes, torch.Tensor)

    return _draw_bounding_boxes(image, boxes, **kwargs)

def postprocess(text, image):
    if image is None:
        return text, None

    colors = ['#ed7d31', '#5b9bd5', '#70ad47', '#7030a0', '#c00000', '#ffff00', "olive", "brown", "cyan"]
    pat = re.compile(r'\[\d*(?:\.\d*)?(?:,\d*(?:\.\d*)?){3}(?:;\d*(?:\.\d*)?(?:,\d*(?:\.\d*)?){3})*\]')
    pat_id = re.compile(r'\<Id(\d+)\>')
    pat_cat = re.compile(r'\]([a-zA-Z0-9_\s]+)\<')
    
    def extract_boxes(string):
        ret = []
        string = string.replace(' ', '')
        for bboxes_str in pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret
        
    def extract_id(string):
        ret = []
        string = string.replace(' ', '')
        for ids_str in pat_id.findall(string):
            ret.append(int(ids_str))
        return ret
    
    def extract_cat(string):
        ret = []
        string = string.replace(' ', '')
        for cats_str in pat_cat.findall(string):
            ret.append(cats_str)
        return ret


    extract_pred = extract_boxes(text)
    labels = None
    # if '<Id' in text and '</Id' in text:
    #     extract_ids = extract_id(text)
    #     extract_cats = extract_cat(text)
    #     labels = []
    #     assert len(extract_ids) == len(extract_cats)
    #     for id, cat in zip(extract_ids, extract_cats):
    #         labels.append(cat + str(id))
    
    boxes_to_draw = []
    color_to_draw = []
    for idx, boxes in enumerate(extract_pred):
        color = colors[idx % len(colors)]
        for box in boxes:
            box = [b / 1000 for b in box]
            boxes_to_draw.append(de_norm_box_xyxy(box, w=image.width, h=image.height))
            color_to_draw.append(color)
    if not boxes_to_draw:
        return text, None

    # boxes_to_draw = [xyxy2cxcywh(xyxy) for xyxy in boxes_to_draw]
    # logits = [1]*len(boxes_to_draw)
    # annotated_frame = annotate(image_source=np.asarray(image), boxes=boxes_to_draw, logits=logits, phrases=labels, img_aspect_ratio='pad')
    # res = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    res = draw_bounding_boxes(image=image, boxes=boxes_to_draw, colors=color_to_draw, width=8, labels=labels)
    res = ToPILImage()(res)

    # post process text color
    location_text = text
    edit_text = list(text)
    bboxes_str = pat.findall(text)
    for idx in range(len(bboxes_str) - 1, -1, -1):
        color = colors[idx % len(colors)]
        boxes = bboxes_str[idx]
        span = location_text.rfind(boxes), location_text.rfind(boxes) + len(boxes)
        location_text = location_text[:span[0]]
        edit_text[span[0]:span[1]] = f'<span style="color:{color}; font-weight:bold;">{boxes}</span>'
    text = "".join(edit_text)
    return text, res

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


def prepare_input(tokenizer, data_args, eval_question, eval_images):
    if data_args.use_im_start_end:
        if '<image>' in eval_question:
            qs = eval_question.replace('<image>', DEFAULT_IM_START_TOKEN + DEFAULT_IM_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + '\n')
        else:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IM_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + '\n' + eval_question
    else:
        qs = DEFAULT_IM_PATCH_TOKEN * 256 + '\n' + eval_question

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer(prompt, return_tensors='pt', max_length=tokenizer.model_max_length, truncation=True).input_ids.cuda()

    images, images_tensors = [], []
    for img_file in eval_images:
        images.append(load_image(img_file))
    
    for img in images:
        images_tensors.append(process_image(
            img, 
            data_args.image_processor,
            data_args.image_size,
            data_args.image_aspect_ratio,
        ).to(torch.bfloat16).cuda())
    # image_tensor = data_args.image_processor.preprocess(image.resize((data_args.image_size, data_args.image_size)), return_tensors='pt')['pixel_values'].to(torch.bfloat16).cuda()

    stop_str = DEFAULT_EOS_TOKEN
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    return input_ids, images_tensors, stopping_criteria


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


def eval(golden_cases=None):
    disable_torch_init()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, tokenizer, data_args = build_model_tokenizer(model_args, data_args, training_args)

    while True:
        image_file = input('Image file:')
        query = input('Query:')

        if ',' in image_file:
            image_files = image_file.split(',')
        else:
            image_files = [image_file]
        
        if query == 'stop':
            break
        
    # for i in range(len(golden_cases)):
    #     case = golden_cases[i]
    #     image_files = case['images']
    #     query = case['question']
    #     print('Question:{}'.format(query))

        for img_file in image_files:
            if not osp.exists(img_file):
                continue
        
        images = []
        for image_file in image_files:
            images.append(load_image(image_file))

        input_ids, image_tensor, stopping_criteria = prepare_input(
            tokenizer, 
            data_args,
            # line['question'] + " Let's think step by step.",
            query,
            image_files
        )

        with torch.inference_mode():
            if data_args.use_beam_search:
                output_ids = model.generate(
                    input_ids,
                    images=[torch.cat([img_tensor for img_tensor in image_tensor], dim=0)],
                    num_beams=5,
                    temperature=0.2,
                    # repetition_penalty=1.0,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
            else:
                output_ids = model.generate(
                    input_ids,
                    images=[torch.cat([img_tensor for img_tensor in image_tensor], dim=0)],
                    do_sample=True,
                    temperature=0.2,
                    # repetition_penalty=1.0,
                    max_new_tokens=1024,
                    # use_cache=True,
                    stopping_criteria=[stopping_criteria])

        outputs = decode(tokenizer, input_ids, output_ids)
        print('Response:{}\n'.format(outputs))
        _, image = postprocess(outputs, images[-1])
        if image is not None:
            save_dir = os.path.join(model_args.model_name_or_path, 'outptus', 'vis')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            image_name = image_files[-1].split('/')[-1][:-4]
            image.save(os.path.join(save_dir, 'case{}_'.format(i) + image_name + '_out' + '.jpg'))


if __name__ == "__main__":
    golden_cases = [{'images':['/data/hypertext/danielyu/datasets/vl/DanceTrack/test/dancetrack0011/img1/00000001.jpg'], 'question':'Detect dancing person and dress.For your response, please adhere to the specified category:[xmin,ymin,xmax,ymax] format.'},
                    {'images':['/data/hypertext/danielyu/datasets/vl/DanceTrack/test/dancetrack0011/img1/00000001.jpg'], 'question': 'What is this girl[193,640,279,943] doing in this image?'},
                    {'images':['/data/hypertext/danielyu/datasets/vl/DanceTrack/test/dancetrack0011/img1/00000002.jpg'], 'question': 'What is the difference between [193, 642, 278, 943] and [592, 659, 684, 939]?'},
                    {'images':['/data/hypertext/danielyu/datasets/vl/DanceTrack/test/dancetrack0011/img1/00000001.jpg','/data/hypertext/danielyu/datasets/vl/DanceTrack/test/dancetrack0011/img1/00000139.jpg'], 'question': 'Given frame1:<image> and frame2:<image>,track person<Id1>Frame1:[193, 640, 279, 943]</Id1> and person <Id5>Frame1:[596, 660, 684, 941]</Id5>.To respond correctly, utilize the specified class<Idi>Frame t:[xmin,ymin,xmax,ymax]</Idi> format.'},
                    {'images':['/data/hypertext/danielyu/datasets/vl/DanceTrack/test/dancetrack0011/img1/00000001.jpg','/data/hypertext/danielyu/datasets/vl/DanceTrack/test/dancetrack0011/img1/00000138.jpg'], 'question': 'Given frame1:<image> and frame2:<image>,track the dancing girl on the left.For the trajectories included in the answer, please use the format Tracki<Idi>Frame t:[xmin,ymin,xmax,ymax]</Idi>.'},
                    {'images':['/data/hypertext/danielyu/datasets/vl/MeViS/valid/JPEGImages/1d906e623692/00002.jpg','/data/hypertext/danielyu/datasets/vl/MeViS/valid/JPEGImages/1d906e623692/00025.jpg','/data/hypertext/danielyu/datasets/vl/MeViS/valid/JPEGImages/1d906e623692/00043.jpg'], 'question': 'Given frame1:<image>,frame2:<image> and frame3:<image>,track the black cat in this video clip.Use the specified Tracki<Idi>Frame t:[xmin,ymin,xmax,ymax]</Idi> format for all trajectories in your reply.'},
                    ]
    
    eval(golden_cases)