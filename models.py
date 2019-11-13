import torch
import torch.nn as nn
import copy
import argparse

class ConvLayer(nn.Module):
    def __init__(self, is_encoder, in_dim, out_dim, kernel, stride, padding=0, slope=0.2):
        super().__init__()
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
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.module = nn.Sequential(self.conv, self.bn, self.leakyrelu) if is_encoder \
            else nn.Sequential(self.deconv, self.bn, self.leakyrelu)

    def forward(self, input):
        output = self.module(input)
        return output


class JointNet(nn.Module):
    def __init__(self, in_dim, latent_dim, width, height,
                 encoder_channel_list, decoder_channel_list,
                 encoder_kernel_list, decoder_kernel_list,
                 encoder_stride_list, decoder_stride_list,
                 encoder_padding_list, decoder_padding_list):
        super().__init__()
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
        self.height = height
        self.width = width
        for i, (c, k, s) in enumerate(zip(self.ecl, self.ekl, self.esl)):
            eid = self.in_dim if i == 0 else self.ecl[i-1]
            self.height = (self.height - k)/s + 1
            self.width = (self.width - k)/s + 1
            self.channel = eid
        self.enc_linear = nn.Linear(int(self.height*self.width*self.channel), latent_dim)
        self.dec_linear = nn.Linear(latent_dim, int(self.height*self.width*self.channel))

        """
        layer definition
        """
        self.style_encoder = list()
        for i, (c, k, s, p) in enumerate(zip(self.ecl, self.ekl, self.esl, self.epl)):
            self.encoder_in_dim = in_dim if i == 0 else self.ecl[i-1]
            self.style_encoder.append(
                ConvLayer(True, self.encoder_in_dim, c, k, s, p))
        self.content_encoder = copy.deepcopy(self.style_encoder)

        self.decoder = list()
        for i, (c, k, s, p) in enumerate(zip(self.dcl, self.dkl, self.dsl, self.dpl)):
            self.decoder_in_dim = latent_dim if i == 0 else self.dcl[i-1]
            self.decoder.append(
                ConvLayer(False, self.decoder_in_dim, c, k, s, p))

    def forward(self, style_input, content_input):
        skip_connection_layer = list()
        skip_connection_layer.append(content_input)
        for i, (cenc, senc) in enumerate(zip(self.content_encoder, self.style_encoder)):
            cenc_out = cenc(content_input) if i == 0 else cenc(cenc_out)
            senc_out = senc(style_input) if i == 0 else senc(senc_out)
            skip_connection_layer.append(cenc_out)

        cenc_out = self._squeeze(cenc_out)
        senc_out = self._squeeze(senc_out)
        store_dim = senc_out.shape
        content_latent = self.enc_linear(cenc_out)
        style_latent = self.enc_linear(senc_out)
        latent = torch.mul(content_latent, style_latent)
        decoder_input = (self.dec_linear(latent)).view(store_dim)

        for i, dec in enumerate(self.decoder):
            sc = skip_connection_layer[-1-i]
            if i == 0:
                decoder_output = dec(decoder_input + sc)
            else:
                decoder_output = dec(decoder_output + sc)

        return decoder_output
    
    def _squeeze(self, input):
        output = input.view(input.shape[0], -1)
        return output

    def _unsqueeze(self, input):
        output = input.unsqueeze(2)
        output = output.unsqueeze(3)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dim', type=int, default=3)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--encoder_channel_list', nargs='+', type=int, default=[16, 16, 16, 16, 16, 16])
    parser.add_argument('--encoder_kernel_list', nargs='+', type=int, default=[3, 3, 3, 3, 3, 3])
    parser.add_argument('--encoder_stride_list', nargs='+', type=int, default=[1, 1, 1, 1, 1, 1])
    parser.add_argument('--encoder_padding_list', nargs='+', type=int, default=[0, 0, 0, 0, 0, 0])
    parser.add_argument('--decoder_channel_list', nargs='+', type=int, default=[16, 16, 16, 16, 16, 16])
    parser.add_argument('--decoder_kernel_list', nargs='+', type=int, default=[3, 3, 3, 3, 3, 3])
    parser.add_argument('--decoder_stride_list', nargs='+', type=int, default=[1, 1, 1, 1, 1, 1])
    parser.add_argument('--decoder_padding_list', nargs='+', type=int, default=[0, 0, 0, 0, 0, 0])
    args = parser.parse_args()

    model = JointNet(args.in_dim, args.latent_dim, args.width, args.height,
                     args.encoder_channel_list, args.decoder_channel_list,
                     args.encoder_kernel_list, args.decoder_kernel_list,
                     args.encoder_stride_list, args.decoder_stride_list,
                     args.encoder_padding_list, args.decoder_padding_list)
    
    content_input = torch.randn(4, 3, 64, 64)
    style_input = torch.randn(4, 3, 64, 64)
    output = model(content_input, style_input)
    print(output.shape)

