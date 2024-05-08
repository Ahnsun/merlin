# 8 machines
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 mmgpt/engine/train/train_flash_attn.py \
  --pair_webdatasets laion2b_10m_6merge \
  --interpair_webdatasets track_1m_v2_2merge+det_5m_v1_en_4merge \
  --model_name_or_path /path/models--lmsys--vicuna-7b-v15 \
  --projector conv \
  --conv_stride 2 \
  --image_size 448 \
  --vision_tower /path/vit-large-patch14-448 \
  --freeze_vision_tower False \
  --freeze_projector False \
  --freeze_lm_model False \
  --bf16 True \
  --output_dir checkpoints/pretrain-clip-large+conv2+vicunav-15-7b-mix20m-v1-trackv2 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 2 \
  --llrd True \
  --learning_rate 5e-5 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --weight_decay 0.05 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --fsdp "shard_grad_op auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --report_to none \
  --image_aspect_ratio resize