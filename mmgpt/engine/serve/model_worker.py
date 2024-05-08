"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
import requests
import torch
import uvicorn

from functools import partial
from threading import Thread
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer

from mmgpt.utils.arguments import *
from mmgpt.model.builder import build_model_tokenizer
from mmgpt.utils.constants import (
    WORKER_HEART_BEAT_INTERVAL, DEFAULT_IMAGE_TOKEN, 
    DEFAULT_IM_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from mmgpt.utils.utils import build_logger, server_error_msg, pretty_print_semaphore
from mmgpt.utils.mm_utils import process_image, process_images, load_image_from_base64, KeywordsStoppingCriteria


limit_model_concurrency = 5
GB = 1 << 30
worker_id = str(uuid.uuid4())[:6]
global_counter = 0
model_semaphore = None


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(self, model_args, data_args):
        self.model, self.tokenizer, data_args = build_model_tokenizer(*parser.parse_args_into_dataclasses())

        self.image_size = data_args.image_size
        self.image_aspect_ratio = data_args.image_aspect_ratio
        self.num_patches = data_args.num_patches
        self.controller_addr = data_args.controller_address
        self.worker_addr = data_args.worker_address
        self.image_processor = data_args.image_processor

        self.worker_id = str(uuid.uuid4())[:6]
        self.logger = build_logger("model_worker", f"model_worker_{worker_id}.log")

        model_paths = model_args.model_name_or_path.split("/")
        if model_paths[-1].startswith('checkpoint-'):
            self.model_name = model_paths[-2] + "_" + model_paths[-1]
        else:
            self.model_name = model_paths[-1]

        self.logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")

        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker, args=(self,))
        self.heart_beat_thread.start()

    def register_to_controller(self):
        self.logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        self.logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                self.logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        num_image_tokens = 0
        if images is not None and len(images) > 0:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")
                
                # images = [
                #     process_image(load_image_from_base64(image), self.image_processor, self.image_size, self.image_aspect_ratio)[0].\
                #         to(self.model.device, dtype=torch.bfloat16)
                #     for image in images
                # ]
                # images = torch.stack(images, dim=0)
                images = process_images(
                    [load_image_from_base64(image) for image in images], 
                    self.image_processor
                ).to(torch.bfloat16).cuda()
                
                replace_token = DEFAULT_IM_PATCH_TOKEN * self.num_patches
                if getattr(self.model, 'use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                num_image_tokens = prompt.count(replace_token) * self.num_patches
            else:
                images = None
            image_args = {"images": [images]}
        else:
            images = None
            image_args = {}
            
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(self.model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)
        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return
        
        thread = Thread(target=self.model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        ))
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    worker = ModelWorker(model_args, data_args)
    uvicorn.run(app, host=data_args.host, port=data_args.port, log_level="info")
