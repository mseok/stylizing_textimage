#!/bin/sh


EPOCH=10
BATCH_SIZE=10
COLOR_PATH="../dataset/Capitals_colorGrad64/train/"
NONCOLOR_PATH="../dataset/Capitals64/train/"
LATENT_DIM=1024
LEARNING_RATE=1e-3
EXPNAME='test'
SAVE_FPATH='results/test'
SAVE_EVERY=10


python3 train/train.py --epoch $EPOCH --batch_size $BATCH_SIZE --color_path $COLOR_PATH --noncolor_path $NONCOLOR_PATH --latent_dim $LATENT_DIM --learning_rate $LEARNING_RATE --expname $EXPNAME --save_every $SAVE_EVERY --gpu

# python3 train/train.py --epoch $EPOCH --batch_size $BATCH_SIZE --color_path $COLOR_PATH --noncolor_path $NONCOLOR_PATH --latent_dim $LATENT_DIM \
# --learning_rate $LEARNING_RATE --expname $EXPNAME --load $LOAD --save_fpath $SAVE_FPATH
