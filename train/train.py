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

    gen_lr = get_learning_rate(gen_optimizer)[0]
    dis_lr = get_learning_rate(dis_optimizer)[0]
            
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

    total_loss = args.lambda_val*gen_loss + gan_loss
    total_loss.backward()

    gen_optimizer.step()
    dis_optimizer.step()

    return total_loss.item(), gen_lr, dis_lr

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


def save_checkpoint(state_dict, epoch, cycle=None):
    directory = 'results/{}/'.format(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'save_{}_{}.pth.tar'.format(epoch, cycle) if cycle else directory + 'save_{}.pth.tar'.format(epoch)
    if cycle:
        state_dict['cycle'] = cycle
    torch.save(state_dict, filename)


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


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
    parser.add_argument('--scheduler',
                        action='store_true',
                        help='to use or not use the lr scheduler')
    parser.add_argument('--schedule_factor',
                        default=0.5,
                        type=float,
                        help='factor value for lr scheduler')
    parser.add_argument('--schedule_patience',
                        default=5,
                        type=int,
                        help='patience value for lr scheduler')
    parser.add_argument('--min_lr',
                        default=0,
                        type=float,
                        help='minimum value of learning rate')
    parser.add_argument('--lambda_val',
                        default=2000,
                        type=float,
                        help='lambda value for adjusting balance between generator loss and GAN loss')
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

    # ADVERSARIAL
    discriminator = Discriminator()
    discriminator_loss = nn.BCELoss()

    real = torch.ones((args.batch_size, 1), dtype=torch.float32, requires_grad=False)
    fake = torch.zeros((args.batch_size, 1), dtype=torch.float32, requires_grad=False)

    if args.load:
        print("=> loading checkpoint 'results_new/{}'".format(args.save_fpath))
        checkpoint = torch.load('results_new/' + args.save_fpath)

        prefix = 'module.'
        n_clip = len(prefix)
        gen = checkpoint['gen_model']
        adapted_gen = {k[n_clip:]: v for k, v in gen.items() if k.startswith(prefix)}
        generator.load_state_dict(adapted_gen)
        # gen_optimizer.load_state_dict(checkpoint['gen_opt'])
        dis = checkpoint['dis_model']
        adapted_dis = {k[n_clip:]: v for k, v in dis.items() if k.startswith(prefix)}
        discriminator.load_state_dict(adapted_dis)
        # dis_optimizer.load_state_dict(checkpoint['dis_opt'])
<<<<<<< HEAD
        loaded_epoch = checkpoint['epoch']
        if checkpoint['cycle']:
            loaded_cycle = checkpoint['cycle']
        if checkpoint['losses']:
            loaded_total_loss = checkpoint['losses']
        if checkpoint['gen_lr'] and checkpoint['dis_lr']:
            gen_lr = checkpoint['gen_lr'] if checkpoint['gen_lr'] > args.min_lr else args.min_lr
            dis_lr = checkpoint['dis_lr'] if checkpoint['dis_lr'] > args.min_lr else args.min_lr

=======
        loaded_epoch = checkpoint['epoch'] + 1
#        loaded_cycle = checkpoint['cycle']
        loaded_cycle = 0
>>>>>>> 2fb3c1593c460ca3c889c9f49a070e0cbbcf7aaf
        print("=> loaded checkpoint '{}'".format(args.save_fpath))

    gen_lr = gen_lr if args.load else args.learning_rate
    dis_lr = dis_lr if args.load else args.learning_rate

    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_optimizer, mode='min', factor=args.schedule_factor, patience=args.schedule_patience, verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dis_optimizer, mode='min', factor=args.schedule_factor, patience=args.schedule_patience, verbose=True)

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
    print("Number of parameters in GENERATOR".format(generator_paramters))
    print("Number of parameters in DISCRIMINATOR".format(discriminator_parameters))

    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

    alphabet_list = 'abcdefghijklmnopqrstuvwxyz'
    alphabet_num = 5

    best_loss = 99999
    loaded_epoch = 0 if not args.load else loaded_epoch
    loaded_cycle = 0 if not args.load else loaded_cycle
    total_loss = [] if not args.load else loaded_total_loss
    with SummaryWriter() as writer:
        for epoch in range(args.epoch):
            if epoch < loaded_epoch:
                continue
            epoch_train_loss = []
            dataset = load_dataset_with_glyph(args)
            if epoch == 0:
                print("number of total cycles: {}".format(len(dataset)))
            cycle_interval = len(dataset) // 10
            for batch_idx, sample_batched in enumerate(dataset):
                if epoch == loaded_epoch:
                    if batch_idx < loaded_cycle:
                        continue
                if len(sample_batched['source']) < args.batch_size:
                    if len(sample_batched['source']) < 16:
                        print("continue called, len:", len(sample_batched['source']))
                        continue
                    temp = len(sample_batched['source']) - len(sample_batched['source'])%8
                    sample_batched['source'] = sample_batched['source'][:temp]
                    sample_batched['glyph'] = sample_batched['glyph'][:temp]

                start_time = time.time()
                target_input = sample_batched['source'].permute(0,3,1,2) # b*3*64*(64*26)
                rand_word = ''.join(random.sample(alphabet_list, alphabet_num))
                position_list = alphabet_position(rand_word)
                source_list = []
                for p in position_list:
                    source_list.append(target_input[:,:,:,64*(p-1):64*p])
                source_input = torch.cat(source_list, dim=3) # b*3*64*(64*5)
                glyph_input = sample_batched['glyph'].permute(0,3,1,2) # b*3*64*(64*26)
                loss, gen_lr, dis_lr = train(generator, discriminator, target_input, source_input,
                                             glyph_input, generator_loss, discriminator_loss,
                                             gen_optimizer, dis_optimizer, real, fake, device,
                                             args)
                total_loss.append(loss)
                epoch_train_loss.append(loss)
                temp_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)

                if batch_idx % cycle_interval == 0:
                    save_checkpoint({
                        'epoch': epoch,
                        'gen_model': generator.state_dict(),
                        'dis_model': discriminator.state_dict(),
                        'gen_opt': gen_optimizer.state_dict(),
                        'dis_opt': discriminator.state_dict(),
                        'losses': total_loss,
                        'gen_lr': gen_lr,
                        'dis_lr': dis_lr
                    }, epoch, batch_idx) 

                end_time = time.time()
                time_interval = end_time - start_time
               
                print("epoch: {}, cycle: {}, loss: {}, dis_lr: {:.4f}, gen_lr: {:.4f}, time: {:.2f}sec".format(epoch, batch_idx, loss, dis_lr, gen_lr, time_interval))

            train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            writer.add_scalar('train/loss', train_loss, epoch)

            save_checkpoint({
                'epoch': epoch,
                'gen_model': generator.state_dict(),
                'dis_model': discriminator.state_dict(),
                'gen_opt': gen_optimizer.state_dict(),
                'dis_opt': discriminator.state_dict(),
                'loss': train_loss
            }, epoch)

            if args.scheduler:
                scheduler.step(train_loss)
            
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
