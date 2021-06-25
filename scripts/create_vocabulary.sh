#!/bin/bash
# Create a vocabulary from a dataset in the COCO-style format.
vocab_name=''
find . -name '*.txt' -print0 | xargs -0 cat | tee -a "../$vocab_name.txt"
yttm bpe --data "$vocab_name.txt" --model "$vocab_name.bpe" --vocab_size "58219" --coverage "1.0"
