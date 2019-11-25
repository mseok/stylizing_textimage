#!/bin/sh

INPUT_LOCATION='source_input.png'
PRETRAINED_LOCATION='results/ckpt.pt'
OUTPUT_FOLDER = '../output/'
OUTPUT_NAME = 'test.png'
LATENT_DIM=1024
COLOR_PATH="../dataset/Capitals_colorGrad64/train/"
NONCOLOR_PATH="../dataset/Capitals64/BASE/"


python3 test/test.py --input_location $INPUT_LOCATION --pretrained_location $PRETRAINED_LOCATION --output_foler $OUTPUT_FOLDER \
                     --output_name $OUTPUT_NAME --latent_dim $LATENT_DIM --color_path $COLOR_PATH --noncolor_path $NONCOLOR_PATH 