import torch
import base64
import requests

from PIL import Image
from io import BytesIO
from transformers import StoppingCriteria


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img) # for simpler box processing
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img) # for simpler box processing
        return result


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


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
    

def process_images(images, image_processor):
    return image_processor(images, return_tensors='pt')['pixel_values']


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


def filter_no_caption_or_no_image(sample):
    return ("txt" in sample or "json" in sample) and ("png" in sample or "jpg" in sample or "jpeg" in sample)
    

def filter_no_text_or_no_image(sample):
    return (b"text_list" in sample['json']) and (b"image_info" in sample['json'])


def filter_size(sample):
    width, height = sample[0].size
    return width > 256 and height > 256


def filter_caption(tokenizer, sample):
    result = tokenizer(sample[1])
    count = 0
    for id in result["input_ids"]:
        if id == 32000:
            count = count + 1
    if count != 1:
        return False
    else:
        return True


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image