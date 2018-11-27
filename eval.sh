#!/usr/bin/env bash
#
# eval.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#


python src/eval.py \
    --attn_model general \
    --embedding_size 256 \
    --hidden_size 256 \
    --n_layers 2 \
    --dropout 0.1 \
    --language afr \
    --input "i love you." \
    --max_len 10 \
    --beam_size 5 \
    --batch_size 1 \
    --device cpu \
    --seed 19 \

/
