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
    parser.add_argument('--epochs',
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
    args = parser.parse_args()

    for batch_idx, (data, target) in enumerate(load_glyph_dataset(args, True)):
        # print (batch_idx)
        data = data.transpose(2,1).transpose(3,2) # B x 64 x (64 x 26) x 3
        # print (data.size())
        position_list = alphabet_position('g')
        glyph_list = []
        for p in position_list:
            glyph_list.append(data[0][:,64*(p-1):64*p,:])
        output_glyph = torch.cat (glyph_list, dim=1) # 이거 갖다 쓰면 되긴함.
        plt.imshow (torch.cat(glyph_list, dim=1))
        if (batch_idx == 1):
            break
    plt.imsave('test1.png', output_glyph)
    plt.imsave('test2.png', output_glyph)
    # print(output_glyph)
    # print(args.epochs)

    # Data loader

    # pretrained model for content selector
    pt = models.vgg16(pretrained=True).features.eval()
    # for source in train_dataset:
        # model, content_losses = transfer_model(pt, source)
    model, content_losses = transfer_model(pt, output_glyph.unsqueeze(0).transpose(1,3))

    # model, content_losses = transfer_model(pt, style_img)
    model(output_glyph.unsqueeze(0).transpose(1,3))
    print(content_losses)
    # content_score = 0
    # for i, sl in enumerate(content_losses):
    #     content_score += 100 * sl.loss
    # content_score
