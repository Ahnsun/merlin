<p align="center">
  <a href="#">
<img src="./assets/merlin_logo.png" alt="Logo" width="130"></a>
  <h1 align="center"><font color="#966661">Merlin</font></h1>
</p>


<h3><a href="">Merlin: Empowering Multimodal LLMs with Foresight Minds</a></h3>

[En Yu](https://ahnsun.github.io/), [Liang Zhao](), [Yana Wei](), [Jinrong Yang](https://yancie-yjr.github.io/), [Dongming Wu](), [Lingyu Kong](), [Haoran Wei](https://scholar.google.com/citations?user=J4naK0MAAAAJ&hl=en), [Tiancai Wang](), [Zheng Ge](https://joker316701882.github.io/), [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en), and [Wenbing Tao]()
	
<a href="https://ahnsun.github.io/merlin/"><img src="https://img.shields.io/badge/Project-Page-Green"></a>
<a href="https://arxiv.org/pdf/2312.00589.pdf"><img src="https://img.shields.io/badge/Paper-PDF-orange"></a> 

Merlin is a groundbreaking model capable of generating natural language responses that are intricately linked with object trajectories of multiple images. Merlin excels in predicting and reasoning about future events based on initial observations, showcasing an unprecedented capability in future prediction and reasoning. Merlin achieves SOTA performance on the established Future Reasoning Benchmark and mulitiple existing MLLM benchmark (MMbench and MMVet), which shows powerful multi-modal general ability and forsight minds.


![](assets/merlin.png)

Code, model weights, and demo will be released soon.

## Release
- [2024/05/06] 🔥🔥🔥We release the source code and weights of Merlin, including training and eval codes.

## Contents
- [Install](#install)
- [Merlin Weights](#merlin-weights)
- [Train](#train)
- [Eval](#eval)

## Install

1. Clone this repository and navigate to the project folder
```bash
git clone https://github.com/Ahnsun/merlin.git
cd /path/to/merlin
```
2. Install Package
```Shell
conda create -n merlin python=3.10 -y
conda activate merlin
pip install e .
```

3. Install Flash-Attention
```
pip install ninja
pip install flash-attn --no-build-isolation
```

## Data

Please download the raw image or video data following the Merlin paper. To define new dataset information, refer to the `merlin/mmgpt/utils/constants.py`. We currently support two types of data reading:

1. Using JSON to store annotations, where JSON files and images are stored locally or on S3.
2. Using Tarfiles to simultaneously store images and annotation information, where the tarfile is stored locally or on S3.

Considering the inefficiency of reading large-scale and complex data from JSON files, and aiming to enhance data supply performance, we **sincerely recommend using only the first type of data for supervised fine-tuning and the second type of data for pretraining.** We are gradually eliminating the usage of JSON data during the pretraining process until all JSON data is exclusively used for supervised fine-tuning.

On top of these two types of data feeds, we support various types of data for online training:

1. **Conversation Data**: We've retained the [Vicuna]/[Llava]-style construction process for conversations, where each round of dialogue is tokenized and concatenated online. Additionally, we provide support for additional boxed data. You need to ensure that each sample includes bounding box coordinates in the "boxes" key, following the format [[x1, y1, h1, w1], ..., [xn, yn, hn, wn]] for all the boxes in the image. **We highly recommend using this type of data for training only during the SFT (Supervised Fine-Tuning) process.**

2. **Image-Text Pair Data**:
This is our **primary pretraining data type**. All the data is preprocessed into **tarfiles** and streamed using the [webdataset]() library. Since the LLM typically encounters the data only once during pretraining, we perform a **weak shuffle** of the data (reading 1000 samples as a local batch randomly from a tarfile each time). Furthermore, to ensure minimal data duplication, we use the InfiniteShardList to read all the tarfiles in a chain.
We have default support for sequence merge logic: Each "getitem" operation directly extracts N image-text pairs and concatenates them into a sequence with an EOS token as a separator, **without separating the attention mask**. This approach maximizes the utilization of LLM's large context length and minimizes data bubbles during training.
After extracting N image-text pair samples, we sequentially tokenize each pair online. We also pre-determine if the current pair would cause a context length overflow. If it does, we discard all subsequent samples.
To support multi-task data training, we allow setting a task prompt for each pair of data and mask the task prompt token during training. This enables us to support single-turn QA conversation data in a similar format for pretraining. For handling more complex multi-turn QA conversations, we have pre-tokenized and organized 22 QA datasets, and **we provide support for reading this pre-tokenized data as well.**

3. **Interpair Data**: We've gone the extra mile to support data types where **multiple images correspond to a single text in video/tracking tasks**. We call this type of data "interleaved pair" (or simply interpair). And yes, this data also supports task prompts (which, in fact, are essential for multi-task training).

4. **Interleave Data**:
To cater to the needs of interactive image-text data with multiple images and segments of text (such as [MMC4](), [OBLISC](), News, and more), we've implemented a **one-to-many** data organization using [Run-Webdataset](). This means that a text list corresponds to all the images in the text.
We've diligently and comprehensively packaged various types of open-source and in-house interleave data into tarfiles. **Interleave data tends to be longer, so we don't provide concatenation for this type of data.** However, in the future, we'll explore more scientific and efficient approaches to data concatenation.


## Merlin Weights
- Download the Merlin weights [here](https://huggingface.co/Kangheng/Merlin). 
- Download the Merlin-Chat weights [here](https://huggingface.co/Kangheng/Merlin-chat). 
- Download the CLIP-VIT-L [here](https://huggingface.co/openai/clip-vit-large-patch14/).

## Framework

Merlin is build based on MMGPT. MMGPT is to be an open-source MultiModal Generative Pretrained Transformers library based on PyTorch and Transformers.

<details open>
<summary>Major features</summary>

- **Module Design**

  We decompose the MMGPT framework into different components and one can easily construct a customized MMGPT framework by combining different modules.

- **Support of various high-performance MMGPTs**

  The library directly includes multiple general understanding frameworks such as **LLava**, **ChatSpot**, **Merlin**.

- **One-click construction of deep and comprehensive benchmark evaluation**

  From mmbench to mmvet, from vqav2 to docvqa, whatever you want!

- **High-performance data provisioning mechanism**

  We have truly broken free from the shackles of ugly and complex low-performance data provisioning tied to JSON. Now, we offer high-performance and high-quality data assurance for a wide range of tasks such as image-text pairs, interleave, VQA (Visual Question Answering), and task prompted QA, spanning from 1,000 to 10,000,000,000 scale.
  
</details>

## Train
```Shell
sh playground/merlin/clip-large+conv+vicuna-v15-7b/pretrain.sh
sh playground/merlin/clip-large+conv+vicuna-v15-7b/sft.sh
```

## Eval
```Shell
sh playground/merlin/clip-large+conv+vicuna-v15-7b/eval.sh
```


## Contact
If you have any questions related to the code or the paper, feel free to email En Yu (`yuen@hust.edu.cn`).

## License
Our model and weights are licensed for both researchers and commercial entities, upholding the principles of openness. The license is drafted by modification of the license of [LLaMA](https://github.com/facebookresearch/llama).

See the [LICENSE](./LICENSE), as well as our accompanying [Acceptable Use Policy](./USE_POLICY.md).

## Citation

If you find our work useful in your research, please consider citing Merlin:
```tex
@article{yuen2023merlin,
  author = {Yu, En and Zhao, Liang and Wei, Yana and Yang, Jinrong and Wu, Dongming and Kong, Lingyu and Wei, Haoran and Wang, Tiancai and Ge, Zheng and Zhang, Xiangyu and Tao, Wenbing},
  title = {Merlin: Empowering Multimodal LLMs with Foresight Minds},
  journal = {arXiv preprint arXiv:2312.00589},
  year = {2023},
}
```
