# stylizing_textimage
> KAIST 2019 Fall, CS470 final project

--- 
## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#Usage)
    - [Train](#Train)
    - [Test](#Test)
    - [Test_non_users](#Test_non_users)
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
> update and install package below
~~~shell
$ pip3 update
$ pip3 install opencv-python, pytorch, ...
~~~
### Datasets
> we use MCGAN datasets, you can get these with below command
</pre>
in datasets/
<code>shell
$ wget https://people.eecs.berkeley.edu/\~sazadi/MCGAN/datasets/Capitals64.tar.gz
$ wget https://people.eecs.berkeley.edu/\~sazadi/MCGAN/datasets/Capitals64.tar.gz
</code></pre>
---
- please extract file on datasets/ directory
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
- Jinwon Lee (??)
- and Yunseok Choi (https://github.com/cys1805)
--- 
## FAQ

- **question?**
    - answer!  
--- 
## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
