#!/usr/bin/env bash
#
# eval.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#


python src/eval.py \
    --attn_model base \
    --embedding_size 1000 \
    --hidden_size 1000 \
    --n_layers 4 \
    --dropout 0.0 \
    --language de \
    --input_file rewrite/test.14.en \
    --output_file test.14.bleu.output \
    --max_len 50 \
    --beam_size 5 \
    --batch_size 1 \
    --device cpu \
    --seed 19
