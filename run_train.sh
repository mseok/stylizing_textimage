#!/bin/sh


EPOCH=20
BATCH_SIZE=10
COLOR_PATH="datasets/Capitals_colorGrad64/train"
LATENT_DIM=1024
LEARNING_RATE=1e-3
EXPNAME='selector'
SAVE_FPATH='selector/save_2.pth.tar'
SAVE_EVERY=500
SCHEDULE_FACTOR=5e-1
SCHEDULE_PATIENCE=5

python3 train/train.py --epoch $EPOCH --load --save_fpath $SAVE_FPATH --batch_size $BATCH_SIZE --color_path $COLOR_PATH --latent_dim $LATENT_DIM --learning_rate $LEARNING_RATE --expname $EXPNAME --save_every $SAVE_EVERY --gpu --scheduler --schedule_factor $SCHEDULE_FACTOR --schedule_patience $SCHEDULE_PATIENCE

