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
    --device cpu \
    --seed 19 \

/
