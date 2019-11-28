#!/bin/sh


EPOCH=50
BATCH_SIZE=48
COLOR_PATH="datasets/Capitals_colorGrad64/train"
LATENT_DIM=1024
LEARNING_RATE=1e-3
EXPNAME='selector'
# SAVE_FPATH='test/save_0_300.pth.tar'
SAVE_EVERY=100

python3 train/train.py --epoch $EPOCH --batch_size $BATCH_SIZE --color_path $COLOR_PATH --latent_dim $LATENT_DIM --learning_rate $LEARNING_RATE --expname $EXPNAME --save_every $SAVE_EVERY --gpu
