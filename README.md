# stylizing_textimage
> KAIST 2019 Fall, CS470 final project

--- 
## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#Usage)
- [Team](#team)
- [FAQ](#FAQ)
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
- If you want more syntax highlighting, format your code like this:
> update and install package below
~~~shell
$ pip3 update
$ pip3 install opencv-python, pytorch, ...
~~~
---
## Usage
### Train
> you can easily use our *run_train.sh*
~~~
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
~~~
- you can set every hyper-parameters here.
- if you want to load pretrained model add **--load** and **--save_fpath** options!
- your training model will save automatically in results/EXPNAME !

### Test
> you can also easily use our *make_font.sh
~~~
#!/bin/sh

INPUT_LOCATION="pocari.png"
SOURCE_CHARACTER='pocar'
PRETRAINED_LOCATION='results/selector/save_9.pth.tar'
OUTPUT_NAME='pocar_font.png'
BATCH_SIZE=512
LATENT_DIM=1024

python3 test/test_users.py --input_location $INPUT_LOCATION --source_character $SOURCE_CHARACTER --pretrained_location $PRETRAINED_LOCATION --output_name $OUTPUT_NAME --batch_size $BATCH_SIZE --latent_dim $LATENT_DIM --gpu
~~~
- if you do not want gpu option just delete **--gpu**
- choose 5 letter font and type location in **INPUT_LOCATION**
- your input size should be *(64,320,3)*
- plz type source character in **SOURCE_CHARACTER**
- output will save in results/OUTPUT_NAME !
---
## Team
> We are team 22 in cs470, KAIST
- Seokhyun Moon (https://github.com/mseok)
- Jinwon Lee (??)
- and Yunseok Choi (https://github.com/cys1805)
--- 
## FAQ

- **question?**
    - answer!  
--- 
## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**