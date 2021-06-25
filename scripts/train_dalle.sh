#!/bin/bash
## run_dalle_pytorch.sh
## Runs dalle-pytorch with DeepSpeed using 16 bit precision.

project_name='gumbelvqgan'
afiaka_bpe='captions/afiaka87_58219.bpe.gz'
user_dir=$USER
vqgan_model_path=vqgan_gumbel_f8_8192.ckpt
vqgan_config_path=vqgan_gumbel_f8_8192.yaml
dataset_directory=/mnt/evo_internal_1TB/CurrentDatasets/COCO

deepspeed train_dalle.py \
	--bpe_path $afiaka_bpe \
	--taming \
	--vqgan_config_path $vqgan_config_path \
	--vqgan_model_path $vqgan_model_path \
	--random_resize_crop_lower_ratio "1.0" \
	--learning_rate '3e-4' \
	--loss_img_weight 1 \
	--attn_types "sparse" \
	--text_seq_len 64 \
	--dim 256 \
	--depth 2 \
	--heads 8 \
	--dim_head 64 \
	--truncate_captions \
	--batch_size 1 \
	--keep_n_checkpoints 5 \
	--image_text_folder "$cache_dir"\dataset \
	--wandb_name "$project_name" \
	--deepspeed \
	--fp16
