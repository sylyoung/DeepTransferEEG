# Pre-training of the mapping module m, i.e. Eq. (4) of the paper
# "Backpropagation-Free Test-Time Adaptation for Lightweight EEG-Based BCIs"
# (IEEE J. Biomed. Health Inform., 2026):
#
#   L_mapping = E_{x~D_synthetic} || m(x_tilde) - pi_tilde ||_1
#
# -slen is K, the number of transformations: 12 for BFT-A, 10 for BFT-D.
# save_checkpoint writes <sodeep>/weights/<name>_best_model.pth.tar, which is
# the name BFT-classify/models/losspredictor.py and
# BFT-regression/train_loss_model.py load.  Run this once per bank size, e.g.
#   python train.py -slen 12 --name 12th_100epochs
#   python train.py -slen 10 --name 10th_100epochs
import argparse

import os
import time
import torch
import torch.nn as nn

from dataset import SeqDataset
from model import model_loader
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from sodeep_utils import AverageMeter, save_checkpoint, log_epoch, count_parameters


device = torch.device("cuda")
# device = torch.device("cpu")


def train(train_loader, model, criterion, optimizer, epoch, print_freq=1):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (s, r) in enumerate(train_loader):

        seq_in, rank_in = s.float().to(device, non_blocking=True), r.float().to(device, non_blocking=True)
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        rank_hat = model(seq_in)
        loss = criterion(rank_hat, rank_in)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), seq_in.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i + 1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses), end="\r")

    print('Train: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
              epoch, i + 1, len(train_loader), batch_time=batch_time,
              data_time=data_time, loss=losses), end="\n")

    return losses.avg, batch_time.avg, data_time.avg


def validate(val_loader, model, criterion, print_freq=1):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (s, r) in enumerate(val_loader):

        seq_in, rank_in = s.float().to(device, non_blocking=True), r.float().to(device, non_blocking=True)
        data_time.update(time.time() - end)

        with torch.set_grad_enabled(False):
            rank_hat = model(seq_in)
            loss = criterion(rank_hat, rank_in)

        losses.update(loss.item(), seq_in.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      i + 1, len(val_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses), end="\r")

    print('Val: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
              i + 1, len(val_loader), batch_time=batch_time,
              data_time=data_time, loss=losses), end="\n")

    return losses.avg, batch_time.avg, data_time.avg


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3, 4, 5, 6, 7'

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("-n", '--name', default="model", help='Name of the model')
    parser.add_argument("-bs", "--batch_size", help="The size of the batches", type=int, default=256)
    parser.add_argument("-lr", dest="lr", help="Initialization of the learning rate", type=float, default=0.001)
    parser.add_argument("-lrs", dest="lr_steps", help="Number of epochs to step down LR", type=int, default=70)
    parser.add_argument("-mepoch", dest="mepoch", help="Max epoch", type=int, default=100)
    parser.add_argument("-pf", dest="print_frequency", help="Number of element processed between print", type=int, default=100)
    parser.add_argument("-slen", dest="seq_len", help="lenght of the sequence process by the ranker", type=int, default=12)
    parser.add_argument("-d", dest="dist", help="index of a single distribution for dataset if None all the distribution will be used.", default=11)
    parser.add_argument('-m', dest="model_type", help="Specify which model to use. (lstm, grus, gruc, grup, exa, lstmla, lstme, mlp, cnn) ", default='lstmla')

    args = parser.parse_args()

    print("Using GPUs: ", os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL'))

    writer = SummaryWriter(os.path.join("./logs/", args.name))

    dset = SeqDataset(args.seq_len, dist=args.dist)

    train_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=2, sampler=SubsetRandomSampler(range(int(len(dset) * 0.1), len(dset))))
    val_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=2, sampler=SubsetRandomSampler(range(int(len(dset) * 0.1))))

    model = model_loader(args.model_type, args.seq_len)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, args.lr_steps, 0.5)

    criterion = nn.L1Loss()      # the L1 norm of Eq. (4)

    print("Nb parameters:", count_parameters(model))

    start_epoch = 0
    best_rec = 10000
    for epoch in range(start_epoch, args.mepoch):
        is_best = False
        lr_scheduler.step()
        train_loss, batch_train, data_train = train(train_loader, model, criterion, optimizer, epoch, print_freq=args.print_frequency)
        val_loss, batch_val, data_val = validate(val_loader, model, criterion, print_freq=args.print_frequency)

        if(val_loss < best_rec):
            best_rec = val_loss
            is_best = True

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_rec': best_rec,
            'args_dict': args
        }

        log_epoch(writer, epoch, train_loss, val_loss, optimizer.param_groups[0]['lr'], batch_train, batch_val, data_train, data_val)
        save_checkpoint(state, is_best, args.name, epoch)

    print('Finished Training')
    print(best_rec)
