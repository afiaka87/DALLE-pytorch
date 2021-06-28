#!/bin/bash

tokenizer=blogoimixer_4096.bpe
data_dir=/workspace/dalle/GumbelDatasets/OI
ckpt_name=oionlyredux
wandb_name=localized_annotations_8kvqgan
log_file=ds_log.txt


deepspeed train_dalle.py \
	--taming \
	--dalle_output_file_name $ckpt_name \
	--bpe_path $tokenizer \
	--wandb_name $wandb_name \
	--image_text_folder $data_dir \
	--clip_grad_norm 1.0 \
	--learning_rate 3e-4 \
	--batch_size 4 \
	--keep_n_checkpoints 5 \
	--save_every_n_steps 1000 \
	--truncate_captions \
	--random_resize_crop_lower_ratio 1.0 \
	--dim 640 \
	--text_seq_len 384 \
	--stable_softmax \
	--depth 8 \
	--heads 16 \
	--dim_head 64 \
	--ff_dropout 0.1 \
	--attn_dropout 0.1 \
	--loss_img_weight 1 \
	--attn_types full,axial_row,axial_col,conv_like \
	--deepspeed --amp --ga_steps 2
