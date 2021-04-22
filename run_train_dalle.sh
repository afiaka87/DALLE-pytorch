#!/bin/bash
# train_dalle.sh
#
#
NUM_GPUS=1
DATASET=$HOME/Datasets
CHECKPOINT=dalle.pt

python train_dalle.py --image_text_folder $DATASET --dalle_path=$CHECKPOINT --taming --truncate_captions