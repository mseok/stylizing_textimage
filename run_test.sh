#!/bin/sh

INPUT_LOCATION="datasets/Capitals_colorGrad64/train/akaPotsley.0.2.png"
PRETRAINED_LOCATION='results/selector/save_0.pth.tar results/selector/save_1.pth.tar results/selector/save_2.pth.tar '
OUTPUT_FOLDER='outputs/'
OUTPUT_NAME='output_test_0.png output_test_1.png output_test_2.png'
LATENT_DIM=1024
COLOR_PATH="datasets/Capitals_colorGrad64/train"
NONCOLOR_PATH="datasets/Capitals64/BASE"
BATCH_SIZE=1

CUDA_VISIBLE_DEVICES=2 python3 test/test.py --input_location $INPUT_LOCATION --pretrained_location $PRETRAINED_LOCATION --output_folder $OUTPUT_FOLDER --output_name $OUTPUT_NAME --latent_dim $LATENT_DIM --color_path $COLOR_PATH --noncolor_path $NONCOLOR_PATH --batch_size $BATCH_SIZE
