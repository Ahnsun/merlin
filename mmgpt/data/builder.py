import torch

from loguru import logger

from mmgpt.data.collator import DataCollatorForSupervisedDataset
from mmgpt.data.dataset.conversation_dataset import ConversationDataset
from mmgpt.data.dataset.pair_webdataset import PairWebDataset
from mmgpt.data.dataset.pair_token_webdataset import PairTokenWebDataset
from mmgpt.data.dataset.interpair_webdataset import InterPairWebDataset
from mmgpt.data.dataset.interleave_webdataset import InterleaveWebDataset

    
def build_dataloader(tokenizer, data_args):
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    multimodal_cfg = dict(
        image_token_len=data_args.image_token_len,
        image_aspect_ratio=data_args.image_aspect_ratio,
        use_im_start_end=data_args.use_im_start_end,
        image_processor=data_args.image_processor,
        box_limit=data_args.box_limit,
        image_size=data_args.image_size
    )

    datasets = []

    if data_args.conversation_datasets:
        conversation_dataset = ConversationDataset(
            tokenizer=tokenizer,
            datasets=data_args.conversation_datasets,
            multimodal_cfg=multimodal_cfg
        )
        datasets.append(conversation_dataset)

    if data_args.pair_webdatasets:
        for dataset in data_args.pair_webdatasets.split('+'):
            pair_webdataset = PairWebDataset(
                tokenizer=tokenizer,
                dataset=dataset,
                multimodal_cfg=multimodal_cfg
            )
            datasets.append(pair_webdataset)

    if data_args.pair_token_webdatasets:
        for dataset in data_args.pair_token_webdatasets.split('+'):
            pair_token_webdataset = PairTokenWebDataset(
                tokenizer=tokenizer,
                dataset=dataset,
                multimodal_cfg=multimodal_cfg
            )
            datasets.append(pair_token_webdataset)

    if data_args.interpair_webdatasets:
        for dataset in data_args.interpair_webdatasets.split('+'):
            interpair_webdataset = InterPairWebDataset(
                tokenizer=tokenizer,
                dataset=dataset,
                multimodal_cfg=multimodal_cfg
            )
            datasets.append(interpair_webdataset)

    if data_args.interleave_webdatasets:
        for dataset in data_args.interleave_webdatasets.split('+'):
            interleave_webdataset = InterleaveWebDataset(
                tokenizer=tokenizer,
                dataset=dataset,
                multimodal_cfg=multimodal_cfg
            )
            datasets.append(interleave_webdataset)

    if len(datasets) == 1 and isinstance(datasets[0], ConversationDataset):
        train_dataset = datasets[0]
    else:
        train_dataset = torch.utils.data.ConcatDataset(datasets)
    logger.info(f'After processing, totally {len(train_dataset)} samples are involved.')
    
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
