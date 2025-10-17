#!/usr/bin/env bash
#
# eval.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#


OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 python src/train.py \
    --attn_model location \
    --embedding_size 1000 \
    --hidden_size 1000 \
    --n_layers 4 \
    --dropout 0.2 \
    --teacher_forcing_ratio 1.0 \
    --clip 5.0 \
    --lr 1 \
    --n_epochs 12 \
    --plot_every 1 \
    --print_every 1 \
    --language de \
    --device cuda \
    --seed 19 \
    --batch_size 128 \
    --reverse True \
