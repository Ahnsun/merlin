import os
import ipdb
import argparse
import torch
import torch.nn.functional as F


def interpolate(path):
    model = torch.load(os.path.join(path, 'pytorch_model-00002-of-00002.bin'), map_location=torch.device('cpu'))

    ipdb.set_trace()
    old_pos_embed = model['model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight']
    old_grid_size = round((old_pos_embed.shape[0] - 1) ** 0.5)
    grid_size = old_grid_size * 2
    extra_tokens = 1

    pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]

    print('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)

    pos_emb_img = pos_emb_img.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=(grid_size, grid_size),
        mode='bicubic',
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size * grid_size, -1)[0]
    new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    model['model.vision_tower.vision_tower.vision_model.embeddings.position_ids'] = torch.arange(1025).expand((1, -1))
    model['model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight'] = new_pos_embed
    torch.save(model, os.path.join(path, 'pytorch_model-00002-of-00002.bin'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    interpolate(args.model_path)

# python mmgpt/utils/interpolate_model.py --model_path /data/hypertext/lucaszhao/MMGPT-PyTorch/checkpoints/pretrain-clip-large+conv1+baichuan2-7b-mix200m-448