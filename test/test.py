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

def make_glyph (args):

    # generator initialize
    generator = Generator (args.latent_dim)

    checkpoint = torch.load(args.pretrained_location)    
    prefix = 'module.'
    n_clip = len(prefix)
    gen = checkpoint['gen_model']
    adapted_gen = {k[n_clip:]: v for k, v in gen.items() if k.startswith(prefix)}
    generator.load_state_dict(adapted_gen)

    source_input = plt.imread(args.input_location)
    print('in test', source_input.shape)
    source_input = torch.from_numpy(source_input).float() # 64*(64*5)*3
    source_input = torch.unsqueeze(source_input.permute(2,0,1), 0) # 1*3*64*(64*5)

    glyph_input = select(args, source_input, input_size=5, source_character='tlqkf') # 1*3*64*(64*26)

    save_image (source_input, 'source_BLOOD.png')
    save_image (glyph_input, 'glyph_BLOOD.png')

    with torch.no_grad():
        return generator (source_input, glyph_input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_location',
                        help='location of input source',
                        type=str,
                        default='source_input.png')
    parser.add_argument('--pretrained_location',
                        help='location of pretrained model',
                        type=str,
                        default='results/ckpt.pt')
    parser.add_argument('--output_folder',
                        help='output folder',
                        type=str,
                        default='../output/')
    parser.add_argument('--output_name',
                        help='location of output png',
                        type=str,
                        default='test.png')
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

    whatwemade = make_glyph(args) # 1*3*64*(64*26)
    whatwemade = torch.squeeze(whatwemade) #.permute(1,2,0) # 64*(64*26)*3
    save_image (whatwemade, 'output_BLOOD.png')
    #cv2.imwrite (args.output_folder + args.output_name, whatwemade.numpy())
    # plt.imsave ('output.png', whatwemade.numpy())
    print ("Congratulations!! output_BLOOD.png saved:)")

    
