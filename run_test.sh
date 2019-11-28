#!/bin/sh

INPUT_LOCATION='BLOOD_tlqkf.png'
PRETRAINED_LOCATION='results/test/save_0_300.pth.tar'
OUTPUT_FOLDER='output/'
OUTPUT_NAME='fuck.png'
LATENT_DIM=1024
COLOR_PATH="datasets/Capitals_colorGrad64/train"
NONCOLOR_PATH="datasets/Capitals64/BASE"
BATCH_SIZE=1

CUDA_VISIBLE_DEVICES=2 python3 test/test.py --input_location $INPUT_LOCATION --pretrained_location $PRETRAINED_LOCATION --output_folder $OUTPUT_FOLDER --output_name $OUTPUT_NAME --latent_dim $LATENT_DIM --color_path $COLOR_PATH --noncolor_path $NONCOLOR_PATH --batch_size $BATCH_SIZE
