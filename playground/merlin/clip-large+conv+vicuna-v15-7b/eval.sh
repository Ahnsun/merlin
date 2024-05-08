# CUDA_VISIBLE_DEVICES=0 torchrun --master_port=43420 mmgpt/engine/eval/eval_mmvet.py \
#     --model_name_or_path ./checkpoints/merlin-chat  \
#     --vision_tower /path/vit-large-patch14-448 \
#     --conv_stride 2 \
#     --image_size 448 \
#     --image_aspect_ratio resize \
#     --projector conv \
#     --bf16 True \
#     --eval_image_path /path/dataset/MM-VET/images/ \
#     --eval_file_path /path/dataset/MM-VET/MMGPT_mm-vet.json \
#     --use_beam_search True \
#     --output_dir ./checkpoints

# CUDA_VISIBLE_DEVICES=1 torchrun --master_port=53421 mmgpt/engine/eval/eval_mmbench.py \
#     --model_name_or_path ./checkpoints/merlin-chat  \
#     --vision_tower /path/vit-large-patch14-448 \
#     --conv_stride 2 \
#     --image_size 448 \
#     --image_aspect_ratio resize \
#     --projector conv \
#     --bf16 True \
#     --eval_file_path /path/dataset/eval_format_files/mmbench_dev_20230712.tsv \
#     --use_beam_search True \
#     --output_dir ./checkpoints

# python -m mmgpt.eval.run_llava_tracking
# --model-name /checkpoints/merlin
# --ngpu 4