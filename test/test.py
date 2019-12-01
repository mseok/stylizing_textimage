import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
import time
import shutil
import cv2
# for pretrained model
import torchvision.models as models

# load models and pretrained selector network
from models.nets import *
from models.selector import *
from data import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision.utils import save_image
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import random


def make_glyph (args):

    # generator initialize
    generator = Generator (args.latent_dim)
    
    output = []
    for path in args.pretrained_location:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        prefix = 'module.'
        n_clip = len(prefix)
        gen = checkpoint['gen_model']
        adapted_gen = {k[n_clip:]: v for k, v in gen.items() if k.startswith(prefix)}
        generator.load_state_dict(adapted_gen)

        target_input = plt.imread(args.input_location) # 64*(64*5)*3
        if (len(target_input.shape)==2):
            target_input = np.stack((target_input,)*3, axis=-1) # gray -> rgb
        target_input = torch.from_numpy(target_input).float() # 64*(64*5)*3
        target_input = torch.unsqueeze(target_input.permute(2,0,1), 0) # 1*3*64*(64*5)

        alphabet_list = 'abcdefghijklmnopqrstuvwxyz'
        alphabet_num = 5

        rand_word = ''.join(random.sample(alphabet_list, alphabet_num))
        position_list = alphabet_position(rand_word)
        source_list = []
        for p in position_list:
            source_list.append(target_input[:,:,:,64*(p-1):64*p])

        source_input = torch.cat(source_list, dim=3) # b*3*64*(64*5)

        glyph_address = args.input_location.replace("_colorGrad64", '64')[:-5] + '0.png'

        glyph_input = plt.imread(glyph_address)
        if (len(glyph_input.shape)==2):
            glyph_input = np.stack((glyph_input,)*3, axis=-1) # gray -> rgb
        glyph_input = torch.from_numpy(glyph_input).float() # 64*(64*26)*3
        glyph_input = torch.unsqueeze(glyph_input.permute(2,0,1), 0) # 1*3*64*(64*26)

        save_image (source_input, 'source_test.png')
        save_image (glyph_input, 'glyph_test.png')

        with torch.no_grad():
            output.append(generator(source_input, glyph_input))

    return torch.squeeze(torch.stack(output, dim=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_location',
                        help='location of input source',
                        type=str,
                        default='source_input.png')
    parser.add_argument('--pretrained_location',
                        help='location of pretrained model',
                        type=str,
                        default='results/ckpt.pt',
                        nargs='+')
    parser.add_argument('--output_folder',
                        help='output folder',
                        type=str,
                        default='../outputs/')
    parser.add_argument('--output_name',
                        help='location of output png',
                        type=str,
                        default='test.png',
                        nargs='+')
    parser.add_argument('--latent_dim',
                        help='latent vector dimension in generator',
                        type=int,
                        default=1024)
    parser.add_argument('--color_path',
                        help='path for style data sources',
                        type=str,
                        default='datasets/Capitals_colorGrad64/train/')
    parser.add_argument('--noncolor_path',
                        help='path for glyph data sources',
                        type=str,
                        default='datasets/Capitals64/BASE/')
    parser.add_argument('--batch_size',
                        help='batch size',
                        type=int,
                        default=1)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_folder)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    whatwemade = make_glyph(args) # 1*3*64*(64*26)
    for idx, opath in enumerate(args.output_name):
       save_image(whatwemade[idx], output_dir + '/' + opath)
       print ("Congratulations!! {} saved:)".format(opath))

    
