import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
import time
import matplotlib.pylab as plt

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
from tensorboardX import SummaryWriter

import torch.utils.data
import torch.utils.data.distributed


def train(model, dataset, criterion, optimizer, epoch, args):
    model.train()

    for data in dataset:
        for i, (input, label) in enumerate(data):
            optimizer.zero_grad()

            ipnut = torch.tensor(
                input, dtype=torch.float32, requires_grad=True)
            label = torch.tensor(label)
            if args.gpu:
                input = input.cuda()
                label = label.cuda()

            pred_label = model(input)
            loss = criterion(pred_label, label)
            loss.backward()
            optimizer.step()


def val():
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',
                        help='number of epochs for training',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='number of batches',
                        type=int,
                        default=1)
    parser.add_argument('--color_path',
                        help='',
                        type=str,
                        default='mini_datasets/Capitals_colorGrad64/')
    parser.add_argument('--noncolor_path',
                        help='',
                        type=str,
                        default='mini_datasets/Capitals64/')
    parser.add_argument('--latent_dim',
                        help='',
                        type=int,
                        default=1024)
    parser.add_argument('--learning_rate',
                        help='',
                        type=float,
                        default=0.001)
    args = parser.parse_args()

    # Data loader
    """
    output_source
    1. input data has colored 26 alphabets 64 * (64 * 26) * 3
    2. get certain position of alphabets via alphabet_position function
    3. get output_source by concating source_list which is selected in (2)
       alphabets from input data(1)
    """
    for batch_idx, (data, _) in enumerate(load_dataset(args, True)):
        data = data # b * 3 * 64 * (64*26)
        position_list = alphabet_position('tlqkf')
        source_list = []
        for p in position_list:
            source_list.append(data[:,:,:,64*(p-1):64*p])
        output_source = torch.cat(source_list, dim=3)

    # pretrained model for content selector
    pt = models.vgg16(pretrained=True).features.eval()
    content_selector, content_losses = transfer_model(pt, output_source)

    content_selector(output_source.transpose(2,3))
    for l in content_losses:
        print(l.loss)
    
    # b * 64 * (64 * 26) * 3 glyph input
    # b * 64 * (64 * 5) * 3 source input
    glyph_input = torch.randn(4, 3, 64, 64*26)
    source_input = torch.randn(4, 3, 64, 64*5)
    target = torch.randn(4, 3, 64, 64*26)
    with SummaryWriter() as writer:
        for epoch in range(args.epoch):
            generator = Generator(args.latent_dim)
            generator_loss = nn.L1Loss()
            gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate)
            # data contain glyph input, source input(5 alphabets), target image
            generated_target = generator(source_input, glyph_input)
            gen_loss = generator_loss(generated_target, target)

            real = torch.ones((generated_target.shape[0], 1), dtype=torch.float32, requires_grad=False)
            fake = torch.zeros((generated_target.shape[0], 1), dtype=torch.float32, requires_grad=False)
            discriminator = Discriminator()
            discriminator_loss = nn.BCELoss()
            dis_optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate)
            gan_loss = discriminator_loss(discriminator(generated_target), fake) \
                     + discriminator_loss(discriminator(target), real)
            
            total_loss = gen_loss + gan_loss
            total_loss.backward()
            gen_optimizer.step()
            dis_optimizer.step()
            print(total_loss)
            
    # train dataloader -> glyph 26, source 5
    # via model -> generate total all 26 alphabets
    # answer -> 26 alphabets same class with source