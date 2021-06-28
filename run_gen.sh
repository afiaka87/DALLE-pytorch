#!/bin/bash

vqgan_model_path=/root/.cache/dalle/vqgan_8192_gumbel_f8_model.ckpt
vqgan_config_path=/root/.cache/dalle/vqgan_8192_gumbel_f8_model.yaml

python generate.py \
    --dalle_path blogoimixer.pt \
    --taming \
    --vqgan_config_path $vqgan_config_path \
    --vqgan_model_path $vqgan_model_path \
    --text "in this image there are people on the left walking on a sidewalk and over here some tree and in the top right a building on the top left a sky" \
    --num_images 1 \
    --batch_size 1 \
    --top_k 0.9 \
    --outputs_dir outputs \
    --bpe_path blogoimixer_4096.bpe \
