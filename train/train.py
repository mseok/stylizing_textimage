import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
import time

# for pretrained model
import torchvision.models as models

# load models and pretrained selector network
from models.nets import *
from models.selector import *

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
    args = parser.parse_args()

    print(args.epochs)

    # Data loader

    # pretrained model for content selector
    # pt = models.vgg16(pretrained=True).features.eval()
    # for source in train_dataset:
    #     model, content_losses = transfer_model(pt, source)

    # model, content_losses = transfer_model(pt, style_img)
    # model(content_img)
    # content_score = 0
    # for i, sl in enumerate(content_losses):
    #     content_score += 100 * sl.loss
    # content_score
