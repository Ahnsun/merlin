import argparse
import ast
# from mmgpt.model.mmgpt import img_token_len_no_pad, input_ids_no_img_pad
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import re
from mmgpt.utils.conversation import conv_templates, SeparatorStyle
from mmgpt.utils.utils import disable_torch_init, expand2square
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from mmgpt.model import *
from mmgpt.utils.utils import KeywordsStoppingCriteria
# from mmgpt.eval.draw_box import xyxy2cxcywh, annotate

from PIL import Image, ImageFont

import os
import requests
from PIL import Image
from io import BytesIO
import os.path as osp
from torchvision.transforms import ToPILImage, PILToTensor
from torchvision.utils import draw_bounding_boxes as _draw_bounding_boxes
from torchvision.ops import box_convert
import numpy as np


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

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

def postprocess(text, images):

    if images is None:
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

    # def extract_boxes(text):

    #     pattern = re.compile(r'([a-zA-Z0-9_\s]+):\[([\d, \[\]]+)\]')
    #     matches1 = pattern.findall(text) #[[label, box],]

    #     pattern = re.compile(r'\[([\d, \[\]]+)\],([a-zA-Z0-9_\s]+)')
    #     matches2 = pattern.findall(text) # box, label   
    #     matches2 = [(j,i) for i, j in matches2]
    #     matches = matches1+matches2

    #     boxes = []
    #     labels = []

    #     for label, box_string in matches:

    #         box_pattern = re.compile(r'\[(\d+), (\d+), (\d+), (\d+)\]')
    #         box_matches = box_pattern.findall('['+box_string+']')

    #         for match in box_matches:
    #             box = [int(coordinate) for coordinate in match]
    #             boxes.append(box)
    #             labels.append(label)


    #     boxes_tensor = torch.tensor(boxes)/1000
    #     # labels_tensor = torch.tensor(labels)

    #     return labels, boxes_tensor

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
    if 'Frame' in text or 'frame' in text:
        if len(extract_pred) > len(images):
            extract_pred = extract_pred[:len(images)]
        for idx, (boxes, image) in enumerate(zip(extract_pred, images)):
            color = colors[idx % len(colors)]
            w = image.width
            h = image.height
            for box in boxes:
                box = [b / 1000 for b in box]
                boxes_to_draw.append(de_norm_box_xyxy(box, w=max(w, h), h=max(w, h)))
                color_to_draw.append(color)
    else:
        for idx, boxes in enumerate(extract_pred):
            color = colors[idx % len(colors)]
            for box in boxes:
                box = [b / 1000 for b in box]
                boxes_to_draw.append(de_norm_box_xyxy(box, w=max(image.width, image.height), h=max(image.width, image.height)))
                color_to_draw.append(color)

    if not boxes_to_draw:
        return text, None
    # boxes_to_draw = [xyxy2cxcywh(xyxy) for xyxy in boxes_to_draw]
    # logits = [1]*len(boxes_to_draw)
    # annotated_frame = annotate(image_source=np.asarray(image), boxes=boxes_to_draw, logits=logits, phrases=labels, img_aspect_ratio='pad')
    # res = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    reses = []
    if 'Frame' in text or 'frame' in text:
        for box, image in zip(boxes_to_draw, images):
            res = draw_bounding_boxes(image=image, boxes=[box], colors=color_to_draw, width=8, labels=labels)
            reses.append(ToPILImage()(res))
    else:
        res = draw_bounding_boxes(image=images[-1], boxes=boxes_to_draw, colors=color_to_draw, width=8, labels=labels)
        res = ToPILImage()(res)
        reses = [res]
    # post process text color
    # location_text = text
    # edit_text = list(text)
    # bboxes_str = pat.findall(text)
    # for idx in range(len(bboxes_str) - 1, -1, -1):
    #     color = colors[idx % len(colors)]
    #     boxes = bboxes_str[idx]
    #     span = location_text.rfind(boxes), location_text.rfind(boxes) + len(boxes)
    #     location_text = location_text[:span[0]]
    #     edit_text[span[0]:span[1]] = f'<span style="color:{color}; font-weight:bold;">{boxes}</span>'
    # text = "".join(edit_text)
    return None, reses


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = MMGPTLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()

    vision_tower = model.get_model().vision_tower
    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower = vision_tower
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)

    # other model params
    model.get_model().image_aspect_ratio = args.image_aspect_ratio
    model.get_model().with_img_padding = args.with_img_padding


    image_processor = CLIPImageProcessor.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16)

    use_im_start_end = getattr(model.config, "use_im_start_end", False)
    print(use_im_start_end)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = use_im_start_end
    if use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    # image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    image_token_len = 256
    # image_file = args.image_file
    # query = args.query

    conv = conv_templates['llava_v1'].copy()
    images = []
    conv_image_files = []
    num_turn = 0

    while True:
        if num_turn == 0:
            image_file = input('Image file:')
        query = input('Query:')
        num_turn += 1

        if ',' in image_file:
            image_files = image_file.split(',')
        else:
            image_files = [image_file]
        
        if image_file != '':
            conv_image_files = image_files

        if query=='stop':
            break
        elif query =='new conv':
            conv = conv_templates['llava_v1'].copy()
            images = []
            conv_image_files = []
            num_turn = 0
            continue

        # for img_file in image_files:
        #     if not osp.exists(img_file):
        #         continue

        # eval per case
        qs = query
        if use_im_start_end:
            if args.task_mode == 'Track':
                qs = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN) * len(image_files) + qs
            elif args.task_mode == 'Detect':
                qs = qs + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
            elif args.task_mode == 'ImgInd':
                qs = qs.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN)
        else:
            qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        prompt_list = prompt.split(conv.roles[0])[1:]
        new_prompt = conv.roles[0] + f'{conv.roles[0]}'.join(prompt_list)    
        inputs = tokenizer([new_prompt])

        if image_files[0] != '':
            for image_file in image_files:
                images.append(load_image(image_file))
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        if args.image_aspect_ratio == 'pad':
            image_size = 448
            image_tensors = []
            for image in images:
                img_size = image.size
                image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
                image_tensors.append(image_processor.preprocess(image, return_tensors='pt', do_center_crop=False,size={"shortest_edge": image_size})['pixel_values'][0])
        
            if args.with_img_padding == False:
                imgs_tokens_len = img_token_len_no_pad(img_size, patch_size=14)
                im_patch_token = getattr(model.config, "im_patch_token", -1)
                im_start_token = getattr(model.config, "im_start_token", -1)
                im_end_token = getattr(model.config, "im_end_token", -1)
    
                input_ids, _ = input_ids_no_img_pad(input_ids, None, [imgs_tokens_len], im_patch_token, im_start_token, im_end_token, use_im_start_end=True, pad_token_id=0, training=False)
        else:
            image_tensors = []
            for image in images:
                image_tensors.append(image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0])
            

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[torch.cat([img_tensor.unsqueeze(0).half().cuda() for img_tensor in image_tensors], dim=0)],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria]
                # img_sizes=[img_size]
                )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        conv.messages[-1][-1] = outputs
        print(outputs)
        _, draw_images = postprocess(outputs, images)
        if draw_images is not None:
            if 'Frame' in outputs or 'frame' in outputs:
                image_names = [image_file.split('/')[-1][:-4] for image_file in conv_image_files]
                for draw_image, image_name in zip(draw_images, image_names):
                    draw_image.save(os.path.join('./figs', 'vis', image_name + '_out_conv{}'.format(num_turn) + '.jpg'))
            else:
                image_name = conv_image_files[-1].split('/')[-1][:-4]
                draw_images[0].save(os.path.join('./figs', 'vis', image_name + '_out_conv{}'.format(num_turn) + '.jpg'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--task-mode", type=str, default='ImgInd')
    parser.add_argument("--image_aspect_ratio", type=str, default='pad')
    parser.add_argument("--with_img_padding", type=ast.literal_eval, default=True)
    args = parser.parse_args()

    eval_model(args)
