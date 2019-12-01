# stylizing_textimage
> KAIST 2019 Fall, CS470 final project

--- 
## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
    - [Clone](#Clone)
    - [Setup](#Setup)
    - [Pretrained models](#Pretrained models)
- [Usage](#Usage)
    - [Train](#Train)
    - [Test](#Test)
    - [Test_non_users](#Test_non_users)
- [Team](#team)
- [License](#license)

---
## Introduction
This is pytorch implementation of [Character Image Synthesis Based on Selected Content and Referenced Style Embedding, 2019](!https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8784736).

---
## Installation
- All the `code` required to get started
### Clone
- Clone this repo to your local machine using below command
~~~shell
$ git clone https://github.com/mseok/stylizing_textimage
~~~
### Setup
> update and install package below
- pytorch 1.3.1 with torchvision
- CUDA 10.1
- Seaborn
- Matplotlib
- OpenCV2
- Numpy
~~~
### Datasets
> we use MCGAN datasets, you can get these with below command you should in main directory

~~~shell
$ mkdir datasets && cd datasets
$ wget https://people.eecs.berkeley.edu/\~sazadi/MCGAN/datasets/Capitals64.tar.gz
$ wget https://people.eecs.berkeley.edu/\~sazadi/MCGAN/datasets/Capitals_colorGrad64.tar.gz
~~~
### Pretrained models
> Due to large size of our pretrained save file, we uploaded it on google drive. You should
execute following command in main directory. **Note that this only works for LINUX system!!**
- our google drive address: https://drive.google.com/drive/folders/1Fq3r8tweP1p2inDDrSbVr-1d-it8EwU6
- How to download save file into shell
    1. Select a file that is need to be downloaded and do right click.
    2. Copy the link for sharing like https://drive.google.com/file/d/1fTX1wt_daS1xWJbRIdq6GkJiWY1EEzoq/view?usp=sharing
    3. Extract file ID like from above 1fTX1wt_daS1xWJbRIdq6GkJiWY1EEzoq
    4. Fill file ID in FILEID and fill download filename in FILENAME in this command: 
~~~shell
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt
~~~
- Following command represent above instructions.
~~~shell
$ mkdir results/ && mkdir results/selector/ && cd results/selector
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fTX1wt_daS1xWJbRIdq6GkJiWY1EEzoq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fTX1wt_daS1xWJbRIdq6GkJiWY1EEzoq" -O save_0.pth.tar && rm -rf /tmp/cookies.txt
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dyLZga_a5v2934sPKKHkTVZ0EyA0zOuq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dyLZga_a5v2934sPKKHkTVZ0EyA0zOuq" -O save_5.pth.tar && rm -rf /tmp/cookies.txt 
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AMa7I5TbgLaNlvs8DZUL29QaZNXlhSgy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AMa7I5TbgLaNlvs8DZUL29QaZNXlhSgy" -O save_9.pth.tar && rm -rf /tmp/cookies.txt
~~~
---
- please extract file in datasets after download .tar files
## Usage
### Train
> you can easily use our *run_train.sh*
<pre>
scripts/run_train.sh
<code>
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
</code></pre>
- you can set every hyper-parameters here.
- if you want to load pretrained model add **--load** and **--save_fpath** options!
- your training model will save automatically in results/EXPNAME !
- if you end with setting type below
~~~shell
$ bash scripts/run_train.sh
~~~

### Test
> you can also easily use our *make_font.sh*
<pre>
scripts/make_font.sh
<code>
#!/bin/sh

INPUT_LOCATION="pocari.png"
SOURCE_CHARACTER='pocar'
PRETRAINED_LOCATION='results/selector/save_9.pth.tar'
OUTPUT_NAME='pocar_font.png'
BATCH_SIZE=512
LATENT_DIM=1024

python3 test/test_users.py --input_location $INPUT_LOCATION --source_character $SOURCE_CHARACTER --pretrained_location $PRETRAINED_LOCATION --output_name $OUTPUT_NAME --batch_size $BATCH_SIZE --latent_dim $LATENT_DIM --gpu
</code></pre>
- if you do not want gpu option just delete **--gpu**
- choose 5 letter font and type location in **INPUT_LOCATION**
- your input size should be *(64,320,3)*
- plz type source character in **SOURCE_CHARACTER**
- output will save in results/OUTPUT_NAME !
- *this phase will takes really really long time.. (because of select phase)*
- if you end with setting type below
~~~shell
$ scripts/bash make_font.sh
~~~

### Test_non_users
> if you can't wait select phase, follow below
<pre>
scripts/run_test.sh
<code>
#!/bin/sh

INPUT_LOCATION="datasets/Capitals_colorGrad64/train/akaPotsley.0.2.png"
PRETRAINED_LOCATION='results/selector/save_0.pth.tar results/selector/save_1.pth.tar results/selector/save_2.pth.tar '
OUTPUT_FOLDER='outputs/'
OUTPUT_NAME='output_test_0.png output_test_1.png output_test_2.png'
LATENT_DIM=1024
COLOR_PATH="datasets/Capitals_colorGrad64/train"
NONCOLOR_PATH="datasets/Capitals64/BASE"
BATCH_SIZE=1

python3 test/test.py --input_location $INPUT_LOCATION --pretrained_location $PRETRAINED_LOCATION --output_folder $OUTPUT_FOLDER --output_name $OUTPUT_NAME --latent_dim $LATENT_DIM --color_path $COLOR_PATH --noncolor_path $NONCOLOR_PATH --batch_size $BATCH_SIZE
</code></pre>
- usage is almost same with test
- but this require the images in datasets/Capitals_colorGrad64/
- you can have multiple location and output_name.
- if you end with setting type below
~~~shell
$ bash scripts/run_test.sh
~~~
---
## Team
> We are team 22 in cs470, KAIST
- Seokhyun Moon (https://github.com/mseok)
- Jinwon Lee (https://github.com/grape-tasting-acid)
- and Yunseok Choi (https://github.com/cys1805)

--- 
## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
