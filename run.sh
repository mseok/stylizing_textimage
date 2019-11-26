#!/bin/sh


EPOCH=50
BATCH_SIZE=32
COLOR_PATH="datasets/Capitals_colorGrad64/"
NONCOLOR_PATH="datasets/Capitals64/BASE"
LATENT_DIM=1024
LEARNING_RATE=1e-3
EXPNAME='test'
SAVE_FPATH='results/test'


python3 train/train.py --epoch $EPOCH --batch_size $BATCH_SIZE --color_path $COLOR_PATH --noncolor_path $NONCOLOR_PATH --latent_dim $LATENT_DIM \
--learning_rate $LEARNING_RATE --expname $EXPNAME --gpu

# python3 train/train.py --epoch $EPOCH --batch_size $BATCH_SIZE --color_path $COLOR_PATH --noncolor_path $NONCOLOR_PATH --latent_dim $LATENT_DIM \
# --learning_rate $LEARNING_RATE --expname $EXPNAME --load $LOAD --save_fpath $SAVE_FPATH
