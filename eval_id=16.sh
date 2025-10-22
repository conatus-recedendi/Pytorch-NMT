#!/usr/bin/env bash
#
# eval.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#


python ./src/eval.py \
    --attn_model location \
    --embedding_size 1000 \
    --hidden_size 1000 \
    --n_layers 4 \
    --dropout 0.2 \
    --language de \
    --input_file ./rewrite/test.14.en \
    --input_ref_file ./rewrite/test.14.de \
    --output_file test.14.hypothesis.id=16.de \
    --max_len 50 \
    --beam_size 12 \
    --batch_size 1 \
    --device cuda \
    --seed 19 \
    --reverse True \
    --input_forward True
