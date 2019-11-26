import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
import time
import shutil
import random

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


def train(generator, discriminator, target, source, glyph, gen_criterion,
          dis_criterion, gen_optimizer, dis_optimizer, real, fake, device,
          args):
    generator.train()
    discriminator.train()

    gen_optimizer.zero_grad()
    dis_optimizer.zero_grad()
            
    if args.gpu:
        glyph = glyph.to(device)
        source = source.to(device)
        target = target.to(device)
        real = real.to(device)
        fake = fake.to(device)

    batch = target.shape[0]
    real = torch.ones((batch, 1), dtype=torch.float32, requires_grad=False).to(device)
    fake = torch.zeros((batch, 1), dtype=torch.float32, requires_grad=False).to(device)

    generated_target = generator(source, glyph)
    gen_loss = gen_criterion(generated_target, target)
    gan_loss = dis_criterion(discriminator(generated_target), fake) \
            + dis_criterion(discriminator(target), real)

    total_loss = gen_loss + gan_loss
    total_loss.backward()

    gen_optimizer.step()
    dis_optimizer.step()

    return total_loss.item()

def val(generator, discriminator, target, source, glyph,
        gen_criterion, dis_criterion, real, fake, args):
    generator.eval()
    discriminator.eval()
            
    if args.gpu:
        glyph = glyph.to(device)
        source = source.to(device)
        target = target.to(device)

    batch = target.shape[0]
    real = torch.ones((batch, 1), dtype=torch.float32, requires_grad=False).to(device)
    fake = torch.zeros((batch, 1), dtype=torch.float32, requires_grad=False).to(device)

    generated_target = generator(source, glyph)
    gen_loss = gen_criterion(generated_target, target)
    gan_loss = dis_criterion(discriminator(generated_target), fake) \
            + dis_criterion(discriminator(target), real)
    
    total_loss = gen_loss + gan_loss

    return total_loss


def save_checkpoint(state_dict, epoch, cycle):
    directory = 'results/{}/'.format(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'save_{}_{}.pth.tar'.format(epoch, cycle)
    torch.save(state_dict, filename)


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
                        action='store_true')
    parser.add_argument('--expname',
                        help='experiment name',
                        type=str,
                        default='test')
    parser.add_argument('--load',
                        help='whether load pretrained file or not',
                        action='store_true')
    parser.add_argument('--save_fpath',
                        help='model parameter saved file path',
                        type=str,
                        default='results/test')
    parser.add_argument('--save_every',
                        help='interval of saving model parameters',
                        type=int,
                        default=10)
    args = parser.parse_args()

    """
    output_source
    1. input data has colored 26 alphabets 64 * (64 * 26) * 3
    2. get certain position of alphabets via alphabet_position function
    3. get output_source by concating source_list which is selected in (2)
       alphabets from input data(1)
    """

    # GENERATOR
    generator = Generator(args.latent_dim)
    generator_loss = nn.L1Loss()
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate)

    # ADVERSARIAL
    discriminator = Discriminator()
    discriminator_loss = nn.BCELoss()
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate)

    real = torch.ones((args.batch_size, 1), dtype=torch.float32, requires_grad=False)
    fake = torch.zeros((args.batch_size, 1), dtype=torch.float32, requires_grad=False)

    if args.load:
        print("=> loading checkpoint '{}'".format(args.save_fpath))
        checkpoint = torch.load(args.save_fpath + '/best.pt')

        generator.load_state_dict(checkpoint['gen_model'])
        gen_optimizer.load_state_dict(checkpoint['gen_opt'])
        discriminator.load_state_dict(checkpoint['dis_model'])
        dis_optimizer.load_state_dict(checkpoint['dis_opt'])
        loaded_epoch = checkpoint['epoch']
        loaded_cycle = checkpoint['cycle']
        print("=> loaded checkpoint '{}'".format(args.save_fpath))

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        generator = generator.to(device)
        generator_loss = generator_loss.to(device)
        discriminator = discriminator.to(device)
        discriminator_loss = discriminator_loss.to(device)
        real = real.to(device)
        fake = fake.to(device)
    print("======= Finished Constructing Models =======")

    generator_paramters = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    discriminator_parameters = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    params = generator_paramters + discriminator_parameters
    print(generator_paramters)
    print(discriminator_parameters)

    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

    alphabet_list = 'abcdefghijklmnopqrstuvwxyz'
    alphabet_num = 5

    best_loss = 99999
    loaded_epoch = 0 if not args.load else loaded_epoch
    loaded_cycle = 0 if not args.load else loaded_cycle
    with SummaryWriter() as writer:
        for epoch in range(args.epoch):
            if epoch < loaded_epoch:
                continue
            epoch_train_loss = []
            dataset = load_dataset(args, color=True)
            if epoch == 0:
                print("number of total cycles: {}".format(len(dataset)))
            for batch_idx, (data, _) in enumerate(dataset):
                if epoch == loaded_epoch:
                    if batch_idx < loaded_cycle:
                        continue
                start_time = time.time()
                target_input = data # b * 3 * 64 * (64*26)
                rand_word = ''.join(random.sample(alphabet_list, alphabet_num))
                position_list = alphabet_position(rand_word)
                source_list = []
                for p in position_list:
                    source_list.append(data[:,:,:,64*(p-1):64*p])
                source_input = torch.cat(source_list, dim=3) # b*3*64*(64*5)
                glyph_input = select(args, source_input, input_size=alphabet_num, source_character=rand_word)
                loss = train(generator, discriminator, target_input, source_input,
                             glyph_input, generator_loss, discriminator_loss,
                             gen_optimizer, dis_optimizer, real, fake, device,
                             args)
                epoch_train_loss.append(loss)
                end_time = time.time()
                time_interval = end_time - start_time
               
                if batch_idx % args.save_every == 0: 
                    save_checkpoint({
                        'epoch': epoch,
                        'cycle': batch_idx,
                        'gen_model': generator.state_dict(),
                        'dis_model': discriminator.state_dict(),
                        'gen_opt': gen_optimizer.state_dict(),
                        'dis_opt': discriminator.state_dict(),
			'loss': loss
                    }, epoch, batch_idx)
                            
                print("epoch: {}, cycle: {}, loss: {}, time: {:.2f}sec".format(epoch, batch_idx, loss, time_interval))

            train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            writer.add_scalar('train/loss', train_loss, epoch)
            
            # val_loss = val(generator, discriminator, val_dataset,
            #                generator_loss, discriminator_loss,
            #                args)
            # writer.add_scalar('val/loss', val_loss.item(), epoch)

            # is_best = val_loss <= best_loss
            # if is_best:
            #     best_loss = val_loss
            
            # save_checkpoint({
            #     'epoch': epoch,
            #     'model': discriminator.state_dict(),
            #     'optimizer': dis_optimizer.state_dict(),
            #     'best_loss': best_loss,
            # }, is_best, 'discriminator')

    # train dataloader -> glyph 26, source 5
    # via model -> generate total all 26 alphabets
    # answer -> 26 alphabets same class with source
