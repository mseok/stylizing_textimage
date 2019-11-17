import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import argparse


class SingleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SingleConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_dim,
                      out_channels=out_dim,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        output = self.layers(x)

        return output


class SingleDeconv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SingleDeconv, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_dim*2,
                                out_channels=out_dim,
                                kernel_size=3,
                                padding=1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input1, input2):
        x = torch.cat([input1, input2], 1)
        output = self.layers(x)

        return output


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EncoderBlock, self).__init__()
        self.layers = nn.Sequential(
            SingleConv(in_dim, out_dim),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        output = self.layers(x)

        return output


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_dim//2,
                                           out_channels=in_dim//2,
                                           kernel_size=2,
                                           stride=2)
        self.conv = SingleConv(in_dim, out_dim)

    def forward(self, input1, input2):
        input1 = self.upsample(input1)
        # B C H W
        diffH = input2.shape[2] - input1.shape[2]
        diffW = input2.shape[3] - input1.shape[3]
        input1 = F.pad(input1, [diffW//2, diffW-diffW//2,
                                diffH//2, diffH-diffH//2])
        output = self.conv(torch.cat([input2, input1], dim=1))

        return output


class PaperEncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel):
        super(PaperEncoderBlock, self).__init__()
        self.pad = kernel // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_dim,
                      out_channels=out_dim,
                      kernel_size=kernel,
                      stride=2,
                      padding=self.pad),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        output = self.layers(x)

        return output


class PaperDecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel):
        super(PaperDecoderBlock, self).__init__()
        self.pad = kernel // 2
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_dim*2,
                               out_channels=out_dim,
                               kernel_size=kernel,
                               stride=2,
                               padding=self.pad,
                               output_padding=1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, input1, input2):
        x = torch.cat([input1, input2], 1)
        output = self.layers(x)

        return output


class EncoderConvBlock(nn.Module):
    def __init__(self, kernel, in_dim, out_dim, slope, pool=True):
        super(EncoderConvBlock, self).__init__()
        self.base_conv = nn.Conv2d(in_channels=in_dim,
                                   out_channels=out_dim,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.additional_conv = nn.Conv2d(in_channels=out_dim,
                                         out_channels=out_dim,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
        self.bn = nn.BatchNorm2d(out_dim)
        self.leakyrelu = nn.LeakyReLU(slope)
        self.relu = nn.ReLU(inplace=True)
        self.pool = pool
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.n_layers = int(kernel // 2)
        self.layers = []
        for layer in range(self.n_layers):
            elem = self.base_conv if layer == 0 else self.additional_conv
            act = self.leakyrelu if layer == self.n_layers - 1 else self.relu
            self.layers.append(elem)
            self.layers.append(self.bn)
            self.layers.append(act)
        if pool:
            self.layers.append(self.maxpool)
        self.module = nn.Sequential(*self.layers)

    def forward(self, input):
        output = self.module(input)

        return output


class DecoderConvBlock(nn.Module):
    def __init__(self, kernel, in_dim, out_dim, slope):
        super(DecoderConvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_dim,
                                         out_channels=in_dim,
                                         kernel_size=2,
                                         stride=2)
        self.conv1 = EncoderConvBlock(kernel, in_dim*2, in_dim, slope, False)
        self.conv2 = EncoderConvBlock(kernel, in_dim, out_dim, slope, False)

    def forward(self, input1, input2):
        # input1: target, input2: skip connection layer
        input1 = self.deconv(input1)
        input2 = self.deconv(input2)
        output = torch.cat([input2, input1], dim=1)
        output = self.conv2(self.conv1(output))

        return output


class FixedConvBlock(nn.Module):
    def __init__(self, is_encoder, in_dim, out_dim, kernel, stride, padding=0, slope=0.2):
        super(FixedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_dim,
                              out_channels=out_dim,
                              kernel_size=kernel,
                              stride=stride,
                              padding=padding)
        self.deconv = nn.ConvTranspose2d(in_channels=in_dim,
                                         out_channels=out_dim,
                                         kernel_size=kernel,
                                         stride=stride,
                                         padding=padding)
        self.bn = nn.BatchNorm2d(out_dim)
        self.leakyrelu = nn.LeakyReLU(slope)
        self.module = nn.Sequential(self.conv, self.bn, self.leakyrelu) if is_encoder \
            else nn.Sequential(self.deconv, self.bn, self.leakyrelu)

    def forward(self, input):
        output = self.module(input)

        return output
