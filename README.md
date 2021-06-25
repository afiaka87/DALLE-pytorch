# Train a dalle model from scratch

This fork is just a working directory for me to make highly opinionated changes specific to my desktop gaming pc.
If you have an RTX 2070 and a debian based linux distro installed this fork is for you!
Otherwise, check upstream at lucidrains/dalle-pytorch

## Download and install dalle-pytorch-afiaka

```sh
git clone https://github.com/afiaka87/dalle-pytorch --single-branch 'working_directory'
cd dalle-pytorch;
python3 -m venv dalle_pytorch_venv;
source dalle_pytorch_venv/bin/activate;
```

## Install Microsoft DeepSpeed/Nvidia Apex/W&B

It may say sparse attention has failed to install; that's okay.

```sh
chmod a+x './scripts/install_deepspeed.sh';
'./scripts/install_deepspeed.sh';
```

## Download a dataset

This dataset is overkill for even a 2 layer dalle. It consists of 1.1 million images generated
by OpenAI for their blog post https://openai.com/blog/dall-e/

```sh
cache_dir=~/.cache/dalle/openai_blog_dataset
mkdir -p $cache_dir/captions $cache_dir/images
wget https://www.dropbox.com/s/a4jx0pe6oc1e5r7/openai_dalle_gen.tar.gz
wget https://www.dropbox.com/s/p0qwhefid4p8q0u/blog_captions.tar.gz
tar xvf openai_dalle_gen.tar.gz --directory $cache_dir/images
```

## Create a file containing all your captions.

```sh
find $cache_dir/captions \
    -type f \
    -name '*.txt' \
    -print0 | xargs -0 -P 0 cat | tee -a all_captions.txt
```

## Train a tokenizer with your file.

English benefits from full coverage usually.

```sh
yttm bpe \
    -data all_captions.txt \
    -model 'model_name.bpe' \
    -coverage 1.0 \
    -vocab_size 58219
```

## Train a dalle on your dataset with your tokenizer.

```sh
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
	--image_text_folder $cache_dir/openai_blog_dataset \
	--wandb_name "$project_name" \
	--deepspeed \
	--fp16
```
