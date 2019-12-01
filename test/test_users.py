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

def make_glyph_selector (args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    # generator initialize
    generator = Generator (args.latent_dim).to(device)
    checkpoint = torch.load(args.pretrained_location, map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['gen_model'])

    source_input = plt.imread(args.input_location)[:,:,:3] # 64*(64*5)*3
    if (len(source_input.shape)==2):
        source_input = np.stack((source_input,)*3, axis=-1) # gray -> rgb
    source_input = torch.from_numpy(source_input).float() # 64*(64*5)*3
    source_input = torch.unsqueeze(source_input.permute(2,0,1), 0).to(device) # 1*3*64*(64*5)

    glyph_input = select (args, source_input, input_size=5, source_character=args.source_character) # 1*3*64*(64*26)

    with torch.no_grad():
        return generator (source_input, glyph_input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_location',
                        help='location of input source',
                        type=str,
                        default='source_input.png')
    parser.add_argument('--source_character',
                        help='character of input source',
                        type=str,
                        default='abcde')
    parser.add_argument('--pretrained_location',
                        help='location of pretrained model',
                        type=str,
                        default='results/selector/save_0_0.pth.tar')
    parser.add_argument('--output_name',
                        help='location of output png',
                        type=str,
                        default='test.png')
    parser.add_argument('--batch_size',
                        help='batch size',
                        type=int,
                        default=10)
    parser.add_argument('--latent_dim',
                        help='latent dimention',
                        type=int,
                        default=1024)
    parser.add_argument('--gpu',
                        help='whether use gpu or not',
                        action='store_true')    
    args = parser.parse_args()

    whatwemade = make_glyph_selector(args) # 1*3*64*(64*26)
    directory = 'outputs/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_image (whatwemade, 'outputs/'+args.output_name)
    print ("Congratulations!! {} saved in outputs:)".format(args.output_name))