#!/bin/sh

EPOCH=20
BATCH_SIZE=10
COLOR_PATH="datasets/Capitals_colorGrad64/train"
LATENT_DIM=1024
LEARNING_RATE=1e-3
EXPNAME='selector'
SAVE_FPATH='selector/save_8_1530.pth.tar'
SCHEDULE_FACTOR=5e-1
SCHEDULE_PATIENCE=5
MIN_LR=0
LAMBDA_VAL=2000

python3 train/train.py --epoch $EPOCH --batch_size $BATCH_SIZE --color_path $COLOR_PATH --latent_dim $LATENT_DIM --learning_rate $LEARNING_RATE --expname $EXPNAME --gpu --scheduler --schedule_factor $SCHEDULE_FACTOR --schedule_patience $SCHEDULE_PATIENCE --min_lr $MIN_LR --lambda_val $LAMBDA_VAL

# if want to load, add below command
# --load --save_fpath $SAVE_FPATH
