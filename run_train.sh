#!/bin/sh


EPOCH=50
BATCH_SIZE=50
COLOR_PATH="datasets/Capitals_colorGrad64/train"
NONCOLOR_PATH="datasets/Capitals64/BASE"
LATENT_DIM=1024
LEARNING_RATE=1e-3
EXPNAME='test_scheduler'
SAVE_FPATH='test/save_0_300.pth.tar'
SAVE_EVERY=100

python3 train/train.py --epoch $EPOCH --batch_size $BATCH_SIZE --color_path $COLOR_PATH --noncolor_path $NONCOLOR_PATH --latent_dim $LATENT_DIM --learning_rate $LEARNING_RATE --expname $EXPNAME --save_every $SAVE_EVERY --gpu --load --save_fpath $SAVE_FPATH --scheduler
