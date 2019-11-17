import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
import time
import matplotlib.pylab as plt
import shutil

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


def train(generator, discriminator, dataset, gen_criterion, dis_criterion,
          gen_optimizer, dis_optimizer, real, fake, args):
    generator.train()
    discriminator.train()

    for data in dataset:
        for i, (glyph_input, source_input, target) in enumerate(data):
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            # glyph_input = torch.tensor(glyph_input, dtype=torch.float32, requires_grad=True)
            # source_input = torch.tensor(source_input, dtype=torch.float32, requires_grad=True)
            # target = torch.tensor(target)
            
            if args.gpu:
                glyph_input = glyph_input.cuda()
                source_input = source_input.cuda()
                target = target.cuda()

            generated_target = generator(source_input, glyph_input)
            gen_loss = gen_criterion(generated_target, target)
            gan_loss = dis_criterion(discriminator(generated_target), fake) \
                    + dis_criterion(discriminator(target), real)
            
            total_loss = gen_loss + gan_loss
            total_loss.backward()
            gen_optimizer.step()
            dis_optimizer.step()

            return total_loss

def val(generator, discriminator, dataset, gen_criterion, dis_criterion, real, fake, args):
    generator.eval()
    discriminator.eval()

    for data in dataset:
        for i, (glyph_input, source_input, target) in enumerate(data):

            # glyph_input = torch.tensor(glyph_input, dtype=torch.float32, requires_grad=True)
            # source_input = torch.tensor(source_input, dtype=torch.float32, requires_grad=True)
            # target = torch.tensor(target)
            
            if args.gpu:
                glyph_input = glyph_input.cuda()
                source_input = source_input.cuda()
                target = target.cuda()

            generated_target = generator(source_input, glyph_input)
            gen_loss = gen_criterion(generated_target, target)
            gan_loss = dis_criterion(discriminator(generated_target), fake) \
                    + dis_criterion(discriminator(target), real)
            
            total_loss = gen_loss + gan_loss

            return total_loss


def save_checkpoint(state_dict, is_best, model_type, filename='checkpoint.pt'):
	directory = 'results/{}/'.format(args.expname)
	if not os.path.exists(directory):
		os.makedirs(directory)
	filename = directory + model_type + '_' + filename
	torch.save(state_dict, filename)
	if is_best:
		shutil.copyfile(filename, 'results/{}/'.format(args.expname) + model_type +  '_model_best.pt')


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
                        help='path for style data sources',
                        type=str,
                        default='mini_datasets/Capitals_colorGrad64/')
    parser.add_argument('--noncolor_path',
                        help='path for glyph data sources',
                        type=str,
                        default='mini_datasets/Capitals64/')
    parser.add_argument('--latent_dim',
                        help='latent vector dimension in generator',
                        type=int,
                        default=1024)
    parser.add_argument('--learning_rate',
                        help='learning rate for optimizer',
                        type=float,
                        default=0.001)
    parser.add_argument('--gpu',
                        help='whether use gpu or not',
                        type=bool,
                        default=False)
    parser.add_argument('--expname',
                        help='experiment name',
                        type=str,
                        default='test')
    parser.add_argument('--load',
                        help='whether load pretrained file or not',
                        type=bool,
                        default=False)
    parser.add_argument('--save_fpath',
                        help='model parameter saved file path',
                        type=str,
                        default='results/test')
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
    
    # b * 64 * (64 * 26) * 3 glyph input
    # b * 64 * (64 * 5) * 3 source input
    glyph_input = torch.randn(4, 3, 64, 64*26)
    source_input = torch.randn(4, 3, 64, 64*5)
    target = torch.randn(4, 3, 64, 64*26)
    print("======= Finished Loading Datasets =======")

    # GENERATOR
    generator = Generator(args.latent_dim)
    generator_loss = nn.L1loss()
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate)

    # ADVERSARIAL
    real = torch.ones((args.batch_size, 1), dtype=torch.float32, requires_grad=False)
    fake = torch.zeros((args.batch_size, 1), dtype=torch.float32, requires_grad=False)
    discriminator = Discriminator()
    discriminator_loss = nn.BCELoss()
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate)

    if args.load:
        print("=> loading checkpoint '{}'".format(args.save_fpath))
        generator_checkpoint = torch.load(args.save_fpath + '/generator_model_best.pt')
        discriminator_checkpoint = torch.load(args.save_fpath + '/discriminator_model_best.pt')
        print("=> loaded checkpoint '{}'".format(args.save_fpath))
        generator.load_state_dict(generator_checkpoint['model'])
        gen_optimizer.load_state_dict(generator_checkpoint['optimizer'])
        discriminator.load_state_dict(discriminator_checkpoint['model'])
        dis_optimizer.load_state_dict(discriminator_checkpoint['optimizer'])

    if args.gpu:
        generator = generator.cuda()
        generator_loss = generator_loss.cuda()
        discriminator = discriminator.cuda()
        discriminator_loss = discriminator_loss.cuda()
    print("======= Finished Constructing Models =======")

    best_loss = 99999
    with SummaryWriter() as writer:
        for epoch in range(args.epoch):
            train_loss = train(generator, discriminator, train_dataset,
                               generator_loss, discriminator_loss,
                               gen_optimizer, dis_optimizer, real, fake, args)
            writer.add_scalar('train/loss', train_loss.item(), epoch)
            
            val_loss = val(generator, discriminator, val_dataset,
                           generator_loss, discriminator_loss,
                           real, fake, args)
            writer.add_scalar('val/loss', val_loss.item(), epoch)

            is_best = val_loss <= best_loss
            if is_best:
                best_loss = val_loss
            
            save_checkpoint({
                'epoch': epoch,
                'model': generator.state_dict(),
                'optimizer': gen_optimizer.state_dict(),
                'best_loss': best_loss,
            }, is_best, 'generator')
            
            save_checkpoint({
                'epoch': epoch,
                'model': discriminator.state_dict(),
                'optimizer': dis_optimizer.state_dict(),
                'best_loss': best_loss,
            }, is_best, 'discriminator')

    # train dataloader -> glyph 26, source 5
    # via model -> generate total all 26 alphabets
    # answer -> 26 alphabets same class with source