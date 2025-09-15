#!/usr/bin/env bash
#
# eval.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#


python src/train.py \
    --attn_model dot \
    --embedding_size 1000 \
    --hidden_size 1000 \
    --n_layers 4 \
    --dropout 0.0 \
    --teacher_forcing_ratio 0.0 \
    --clip 5.0 \
    --lr 1 \
    --n_epochs 10 \
    --plot_every 1 \
    --print_every 1 \
    --language spa \
    --device cuda \
    --seed 19
