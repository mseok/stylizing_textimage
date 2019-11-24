#!/bin/sh


EPOCH=10
BATCH_SIZE=10
COLOR_PATH="mini_datasets/Capitals_colorGrad64/"
NONCOLOR_PATH="mini_datasets/Capitals64/BASE"
LATENT_DIM=1024
LEARNING_RATE=1e-3
EXPNAME='test'
SAVE_FPATH='results/test'


python3 train/train.py --epoch $EPOCH --batch_size $BATCH_SIZE --color_path $COLOR_PATH --noncolor_path $NONCOLOR_PATH --latent_dim $LATENT_DIM \
--learning_rate $LEARNING_RATE --expname $EXPNAME

# python3 train/train.py --epoch $EPOCH --batch_size $BATCH_SIZE --color_path $COLOR_PATH --noncolor_path $NONCOLOR_PATH --latent_dim $LATENT_DIM \
# --learning_rate $LEARNING_RATE --expname $EXPNAME --load $LOAD --save_fpath $SAVE_FPATH