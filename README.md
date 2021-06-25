## Quick Start

```sh
#!/bin/bash
# Download, install, and train dalle-pytorch-afiaka
git clone https://github.com/afiaka87/dalle-pytorch --single-branch 'working_directory'
cd dalle-pytorch;
sudo apt-get install python3.9 python3.9-dev python3.9-venv
sudo update-alternatives --config python # (Optional)
python3.9 -m venv dalle_pytorch_venv;
source

```

## Sparse Attention

By default `DALLE` will use full attention for all layers, but you can specify the attention type per layer as follows.

- `full` full attention

- `axial_row` axial attention, along the rows of the image feature map

- `axial_col` axial attention, along the columns of the image feature map

- `conv_like` convolution-like attention, for the image feature map

The sparse attention only applies to the image. Text will always receive full attention, as said in the blogpost.

## Training

## Training using an Image-Text-Folder

```python
$ deepspeed train_dalle.py --image_text_folder "http://storage.googleapis.com/nvdata-openimages/openimages-train-{000000..000554}.tar" --wds jpg,json --taming --truncate_captions --random_resize_crop_lower_ratio=0.8 --attn_types=full --epochs=2 --fp16 --deepspeed
```

In order to convert your image-text-folder to WebDataset format, you can make use of one of several methods.
(https://www.youtube.com/watch?v=v_PacO-3OGQ here are given 4 examples, or a little helper script which also supports splitting your dataset
into shards of .tar.gz files https://github.com/robvanvolt/DALLE-datasets/blob/main/wds_create_shards.py)

### DALL-E with Taming Transformer's VQVAE

Just use the `--taming` flag. Highly recommended you use this VAE over the OpenAI one!

```python
$ python train_dalle.py --image_text_folder /path/to/coco/dataset --taming
```

### Generation

Once you have successfully trained DALL-E, you can then use the saved model for generation!

```python
$ python generate.py --dalle_path ./dalle.pt --text 'fireflies in a field under a full moon'
```

You should see your images saved as `./outputs/{your prompt}/{image number}.jpg`

To generate multiple images, just pass in your text with '|' character as a separator.

ex.

```python
$ python generate.py --dalle_path ./dalle.pt --text 'a dog chewing a bone|a cat chasing mice|a frog eating a fly'
```

### Docker

You can use a docker container to make sure the version of Pytorch and Cuda are correct for training DALL-E. <a href="https://docs.docker.com/get-docker/">Docker</a> and <a href='#'>Docker Container Runtime</a> should be installed.

To build:

```bash
docker build -t dalle-afiaka docker
```

To run in an interactive shell:

```bash
docker run --gpus all -it --mount src="$(pwd)",target=/workspace/dalle-afiaka,type=bind dalle-afiaka:latest bash
```

### Distributed Training

#### DeepSpeed

```sh
$ deepspeed <file>.py [args...] --deepspeed
```

Modify the `deepspeed_config` dictionary in `train_dalle.py` or
`train_vae.py` according to the DeepSpeed settings you'd like to use
for each one. See the [DeepSpeed configuration
docs](https://www.deepspeed.ai/docs/config-json/) for more
information.

#### DeepSpeed - 32 and 16 bit Precision
As of DeepSpeed version 0.3.16, ZeRO optimizations can be used with
single-precision floating point numbers. If you are using an older
version, you'll have to pass the `--fp16` flag to be able to enable
ZeRO optimizations.


#### Custom Tokenizer

The only requirement is that you use `0` as the padding during tokenization
```bash
$ pip install youtokentome

```
#### Chinese

You can train with a <a href="https://huggingface.co/bert-base-chinese">pretrained chinese tokenizer</a> offered by Huggingface 🤗 by simply passing in an extra flag `--chinese`

ex.

```sh
$ python train_dalle.py --chinese --image_text_folder ./path/to/data
```

Then you need to prepare a big text file that is a representative sample of the type of text you want to encode. You can then invoke the `youtokentome` command-line tools. You'll also need to specify the vocab size you wish to use, in addition to the corpus of text.

```bash
$ yttm bpe --vocab_size 8000 --data ./path/to/big/text/file.txt --model ./path/to/bpe.model
```

That's it! The BPE model file is now saved to `./path/to/bpe.model` and you can begin training!

ex.

```sh
$ deepspeed train_dalle.py \
    # ...
    --image_text_folder ./path/to/data \
    --bpe_path ./path/to/bpe.model \
```

```sh
$ python generate.py --chinese --text '追老鼠的猫'
```

## Citations

```bibtex
@misc{ramesh2021zeroshot,
    title   = {Zero-Shot Text-to-Image Generation}, 
    author  = {Aditya Ramesh and Mikhail Pavlov and Gabriel Goh and Scott Gray and Chelsea Voss and Alec Radford and Mark Chen and Ilya Sutskever},
    year    = {2021},
    eprint  = {2102.12092},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```

```bibtex
@misc{kitaev2020reformer,
    title   = {Reformer: The Efficient Transformer},
    author  = {Nikita Kitaev and Łukasz Kaiser and Anselm Levskaya},
    year    = {2020},
    eprint  = {2001.04451},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@misc{esser2021taming,
    title   = {Taming Transformers for High-Resolution Image Synthesis},
    author  = {Patrick Esser and Robin Rombach and Björn Ommer},
    year    = {2021},
    eprint  = {2012.09841},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{ding2021cogview,
    title   = {CogView: Mastering Text-to-Image Generation via Transformers},
    author  = {Ming Ding and Zhuoyi Yang and Wenyi Hong and Wendi Zheng and Chang Zhou and Da Yin and Junyang Lin and Xu Zou and Zhou Shao and Hongxia Yang and Jie Tang},
    year    = {2021},
    eprint  = {2105.13290},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```