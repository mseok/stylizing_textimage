#!/bin/sh

INPUT_LOCATION="pocari.png"
SOURCE_CHARACTER='pocar'
PRETRAINED_LOCATION='results/selector/save_0_0.pth.tar'
OUTPUT_NAME='pocar_font.png'
BATCH_SIZE=10
LATENT_DIM=1024

python3 test/test_users.py --input_location $INPUT_LOCATION --source_character $SOURCE_CHARACTER --pretrained_location $PRETRAINED_LOCATION --output_name $OUTPUT_NAME --batch_size $BATCH_SIZE --latent_dim $LATENT_DIM --gpu