#!/usr/bin/env bash
dataset_dir=/mnt/evo_internal_1TB/CurrentDatasets/OPENIMAGESLOCALIZEDANNOTATIONS
wandb_project=localized_annotations_8kvqgan
python my_train_dalle.py --image_text_folder $dataset_dir --taming --depth 2 --heads 8 --dim 128 --text_seq_len 128
