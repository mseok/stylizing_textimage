import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import argparse

from .layers import *


class CustomJointNet(nn.Module):
    def __init__(self, in_dim, latent_dim, width, height, slope,
                 encoder_kernel_list, decoder_kernel_list):
        super(CustomJointNet, self).__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.width = width
        self.height = height
        self.slope = slope
        self.ekl = encoder_kernel_list
        self.dkl = decoder_kernel_list
        self.style_encoder = []
        self.content_encoder = []
        for k in self.ekl:
            self.style_encoder.append(EncoderConvBlock(
                k, self.in_dim, self.in_dim*2, self.slope))
            self.content_encoder.append(EncoderConvBlock(
                k, self.in_dim, self.in_dim*2, self.slope))
            self.in_dim *= 2
            self.width //= 2
            self.height //= 2

        self.encoder_linear = nn.Linear(
            self.in_dim*self.width*self.height*2, self.latent_dim)
        self.decoder_linear = nn.Linear(
            self.latent_dim, self.in_dim*self.width*self.height)

        self.decoder = []
        for k in self.dkl:
            self.decoder.append(DecoderConvBlock(
                k, self.in_dim, self.in_dim//2, self.slope))
            self.in_dim //= 2
            self.width *= 2
            self.height *= 2

    def forward(self, style_input, content_input):
        sc_outputs = []
        for i, (sl, cl) in enumerate(zip(self.style_encoder, self.content_encoder)):
            if i == 0:
                style_output = sl(style_input)
                content_output = cl(content_input)
            else:
                style_output = sl(style_output)
                content_output = cl(content_output)
            sc_outputs.append(content_output)

        b, c, h, w = content_output.shape
        style_encoder_output = style_output.view(b, -1)
        content_encoder_output = content_output.view(b, -1)
        encoder_output = torch.cat(
            (style_encoder_output, content_encoder_output), -1)
        latent = self.encoder_linear(encoder_output)
        decoder_input = self.decoder_linear(latent)
        decoder_input = decoder_input.view(b, c, h, w)
        for i, dl in enumerate(self.decoder):
            if i == 0:
                output = dl(decoder_input, sc_outputs[-1-i])
            else:
                output = dl(output, sc_outputs[-1-i])

        return output


class PaperJointNet(nn.Module):
    def __init__(self, in_dim, latent_dim, width, height,
                 encoder_channel_list, decoder_channel_list,
                 encoder_kernel_list, decoder_kernel_list,
                 encoder_stride_list, decoder_stride_list,
                 encoder_padding_list, decoder_padding_list):
        super(PaperJointNet, self).__init__()
        self.in_dim = in_dim
        self.ecl = encoder_channel_list
        self.ekl = encoder_kernel_list
        self.esl = encoder_stride_list
        self.epl = encoder_padding_list
        self.ld = latent_dim
        self.dcl = decoder_channel_list
        self.dkl = decoder_kernel_list
        self.dsl = decoder_stride_list
        self.dpl = decoder_padding_list
        self.init_width = width
        self.init_height = height
        self.height = height
        self.width = width
        for i, (c, k, s) in enumerate(zip(self.ecl, self.ekl, self.esl)):
            self.height = int((self.height - k)/s + 1)
            self.width = int((self.width - k)/s + 1)
            self.channel = self.ecl[i]
        self.enc_linear = nn.Linear(
            int(self.height*self.width*self.channel)*2, latent_dim)
        self.dec_linear = nn.Linear(latent_dim, int(
            self.height*self.width*self.channel))

        """
        layer definition
        """
        self.style_encoder = list()
        for i, (c, k, s, p) in enumerate(zip(self.ecl, self.ekl, self.esl, self.epl)):
            self.encoder_in_dim = in_dim if i == 0 else self.ecl[i-1]
            self.style_encoder.append(
                FixedConvBlock(True, self.encoder_in_dim, c, k, s, p))
        self.content_encoder = copy.deepcopy(self.style_encoder)

        self.decoder = list()
        for i, (c, k, s, p) in enumerate(zip(self.dcl, self.dkl, self.dsl, self.dpl)):
            self.decoder_in_dim = self.ecl[-1] if i == 0 else self.dcl[i-1]
            self.decoder.append(
                FixedConvBlock(False, self.decoder_in_dim, c, k, s, p))

    def forward(self, style_input, content_input):
        base = copy.deepcopy(style_input)
        skip_connection_layer = list()
        skip_connection_layer.append(content_input)
        for i, (cenc, senc) in enumerate(zip(self.content_encoder, self.style_encoder)):
            cenc_out = cenc(content_input) if i == 0 else cenc(cenc_out)
            senc_out = senc(style_input) if i == 0 else senc(senc_out)
            skip_connection_layer.append(cenc_out)

        store_dim = senc_out.shape
        cenc_out = self._squeeze(cenc_out)
        senc_out = self._squeeze(senc_out)
        latent = torch.cat((cenc_out, senc_out), 1)
        latent = self.enc_linear(latent)
        decoder_input = (self.dec_linear(latent)).view(store_dim)

        for i, dec in enumerate(self.decoder):
            sc = skip_connection_layer[-1-i]
            if i == 0:
                decoder_output = dec(decoder_input + sc)
            else:
                if decoder_output.shape[2] > sc.shape[2]:
                    sc = self._pad_upsample(sc, decoder_output)
                decoder_output = dec(decoder_output + sc)

        _, final_channel, final_width, __ = decoder_output.shape
        if self.init_width > final_width:
            decoder_output = self._pad_upsample(decoder_output, base)
        else:
            diff = final_width - self.init_width
            conv = nn.Conv2d(final_channel, final_channel, diff + 1)
            decoder_output = conv(decoder_output)

        return decoder_output

    def _squeeze(self, x):
        output = x.view(x.shape[0], -1)

        return output

    def _pad_upsample(self, base, target):
        _, bc, bw, bh = base.shape
        _, __, tw, th = target.shape
        diffw = tw - bw
        diffh = th - bh
        padding = [diffh//2, diffh//2, diffw//2, diffw//2]
        base = F.pad(base, padding, "constant", 0)

        return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dim', type=int, default=3)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--slope', type=float, default=0.2)
    parser.add_argument('--encoder_channel_list', nargs='+',
                        type=int, default=[16, 16, 16, 16, 16, 64])
    parser.add_argument('--encoder_kernel_list', nargs='+',
                        type=int, default=[3, 3, 3, 3, 3, 5])
    parser.add_argument('--encoder_stride_list', nargs='+',
                        type=int, default=[1, 1, 1, 1, 1, 2])
    parser.add_argument('--encoder_padding_list', nargs='+',
                        type=int, default=[0, 0, 0, 0, 0, 0])
    parser.add_argument('--decoder_channel_list', nargs='+',
                        type=int, default=[64, 16, 16, 16, 16, 16])
    parser.add_argument('--decoder_kernel_list', nargs='+',
                        type=int, default=[5, 3, 3, 3, 3, 3])
    parser.add_argument('--decoder_stride_list', nargs='+',
                        type=int, default=[2, 1, 1, 1, 1, 1])
    parser.add_argument('--decoder_padding_list', nargs='+',
                        type=int, default=[0, 0, 0, 0, 0, 0])
    args = parser.parse_args()

    # model = PaperJointNet(args.in_dim, args.latent_dim, args.width, args.height,
    #                       args.encoder_channel_list, args.decoder_channel_list,
    #                       args.encoder_kernel_list, args.decoder_kernel_list,
    #                       args.encoder_stride_list, args.decoder_stride_list,
    #                       args.encoder_padding_list, args.decoder_padding_list)
    model = CustomJointNet(args.in_dim, args.latent_dim, args.width, args.height,
                           args.slope, args.encoder_kernel_list, args.decoder_kernel_list)

    content_input = torch.randn(4, 3, 64, 64)
    style_input = torch.randn(4, 3, 64, 64)
    output = model(style_input, content_input)
    print(output.shape)
    """
    PaperJointNet
    python ./models/models.py --encoder_channel_list 64 128 256 --encoder_kernel_list 7 5 5 --encoder_stride_list 2 2 2 --encoder_padding_list 0 0 0 --decoder_channel_list 128 64 3 --decoder_kernel_list 5 7 7 --decoder_stride_list 2 2 2 --decoder_padding_list 0 0 0
    CustomJointNet
    python ./models/models.py --encoder_kernel_list 7 5 5 --decoder_kernel_list 5 7 7
    """
    # ptmodel = models.vgg16(pretrained=True)
