#!/bin/bash
# Don't run this if you don't know what you're doing.

vocab_name='afiaka87'

#cd <redacted>/CurrentDatasets
#find . -name '*.txt' -print0 | xargs -0 cat | tee -a "../$vocab_name.txt"

# Train a youtokentome byte-pair-encoding from it.
#Usage: yttm bpe [OPTIONS]
#
#  Train BPE model.
#
#Options:
#  --data PATH           Training data file path.  [required]
#  --model PATH          Output model file path.  [required]
#  --vocab_size INTEGER  Number of tokens in the final vocabulary.  [required]
#  --coverage FLOAT      Percentage of characters covered by the model.
#                        [default: 1.0]
#  --n_threads INTEGER   Number of threads.  [default: -1]
#  --pad_id INTEGER      Padding token id.  [default: 0]
#  --unk_id INTEGER      Unknown token id.  [default: 1]
#  --bos_id INTEGER      Begin of sentence token id.  [default: 2]
#  --eos_id INTEGER      End of sentence token id.  [default: 3]
#  --help                Show this message and exit.
#
yttm bpe --data "$vocab_name.txt" --model "$vocab_name.bpe" --vocab_size "58219" --coverage "1.0"
