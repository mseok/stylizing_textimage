import torch
import torch.nn as nn
import argparse
import cv2
# for pretrained model
import torchvision.models as models
import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from data import *
import numpy as np
import matplotlib.pylab as plt


# import data.data_loader

import copy

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # the feature tensor of the style image

    def forward(self, input):
        target_gram = self._gram_matrix(self.target)
        input_gram = self._gram_matrix(input)
        loss_function = nn.MSELoss()
        self.loss = loss_function(input_gram, target_gram)

        return input

    def _gram_matrix(self, input):
        a, b, c, d = input.shape
        input = input.view(input.shape[0] * input.shape[1], -1)
        G = torch.mm(input, input.t())

        return G/(a*b*c*d)


def transfer_model(pretrained_model, content_img):
    pretrained_model = copy.deepcopy(pretrained_model)
    content_losses = []
    model = nn.Sequential()
    
    i = 0
    for layer in pretrained_model.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError("Unrecognized layer: {}".format(layer.__class__.__name__))

        model.add_module(name, layer)
        if name in ['conv_1', 'conv_3', 'conv_5', 'conv_8', 'conv_11']:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss):
            break
        
    model = model[:(i+1)]

    return model, content_losses

def get_layer_info (pretrained_model, content_img):
    pretrained_model = copy.deepcopy(pretrained_model)

# source input : 1*3*64*(64*5), b*3*64*(64*5) -> output: b*1
def _get_loss (source_input, glyph):
    b = glyph.size()[0]
    loss_list = []
    for idx in range(b):
        pt = models.vgg16(pretrained=True).features.eval()
        content_selector, content_losses = transfer_model(pt, torch.unsqueeze(glyph[idx,:,:,:], 0))
        content_selector(source_input)
        loss = 0
        for cl in content_losses:
            loss += cl.loss
        loss_list.append(loss.item())
    return loss_list

def select (source_input, input_size=5, source_character= 'abcde'):
    min_loss = 9999
    selected_glyph = torch.rand(1,3,64,64*5)
    temp_l = []
    for batch_idx, (data, _) in enumerate(load_dataset(args, color=False)):
        position_list = alphabet_position(source_character)
        glyph_list = []
        for p in position_list:
            glyph_list.append(data[:,:,:,64*(p-1):64*p])
        temp_glyph = torch.cat (glyph_list, dim=3)

        temp_l = _get_loss (source_input, temp_glyph)

        if min(temp_l) < min_loss:
            min_idx = temp_l.index(min(temp_l))
            selected_glyph = torch.unsqueeze(data[min_idx,:,:,:], 0)

        if (batch_idx == 0):
            break
    
    return selected_glyph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',
                        help='number of epochs for training',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='number of batches',
                        type=int,
                        default=2)
    parser.add_argument('--color_path',
                        help='path for style data sources',
                        type=str,
                        default='mini_datasets/Capitals_colorGrad64/')
    parser.add_argument('--noncolor_path',
                        help='path for glyph data sources',
                        type=str,
                        default='mini_datasets/Capitals64/')
    args = parser.parse_args()

    selected_glyph = select(torch.rand((1, 3, 64, 64*5)))

    plt.imshow (torch.squeeze(selected_glyph).permute(1,2,0))
    plt.show()

