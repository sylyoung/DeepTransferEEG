# -*- coding: utf-8 -*-
# @Author  : Siyang Li
# @File    : bft.py
# "Backpropagation-Free Test-Time Adaptation for Lightweight EEG-Based
#  Brain-Computer Interfaces" (BFT).  arXiv: https://arxiv.org/abs/2601.07556
#
# BFT adapts each prediction using only forward passes of a fixed source model.
# For every test trial it builds K transformed views, scores each view with a
# reliability ranker r trained once on source data, and aggregates the K
# softmax predictions weighted by their reliability. No gradients touch the
# backbone at test time. Two transformation banks are provided:
#   BFT-A: K knowledge-guided input augmentations.
#   BFT-D: K deterministic feature-masked subnetworks.
# The ranker r is trained on source subjects with a differentiable
# rank-correlation loss, so that its predicted per-view reliabilities agree in
# rank with the real per-view task losses. This single file uses a self-contained
# soft-Spearman loss in place of the external SoDeep sorter used in the paper.
import numpy as np
import argparse
import os
import sys
import time
import gc
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from scipy.signal import hilbert
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import accuracy_score

from utils.network import backbone_net
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, data_loader
from utils.alg_utils import EA_online


class ReliabilityRanker(nn.Module):
    # r(.): maps a backbone feature vector to a scalar predicted task loss.
    # A lower predicted loss means a more reliable view.
    def __init__(self, feature_dim):
        super(ReliabilityRanker, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.ELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4), nn.ELU(),
            nn.Linear(feature_dim // 4, 1))

    def forward(self, x):
        return self.net(x)


def soft_rank(x, tau=1.0):
    # Differentiable rank of each entry of a 1-D score vector, via pairwise
    # sigmoid comparisons. rank_i grows with x_i.
    diff = (x.unsqueeze(1) - x.unsqueeze(0)) / tau
    return torch.sigmoid(diff).sum(dim=1)


def soft_spearman_loss(pred, target, tau=1.0):
    # 1 - Spearman rank correlation between predicted and target reliabilities.
    # Self-contained stand-in for the SoDeep differentiable sorter m(.).
    rp = soft_rank(pred, tau)
    rt = soft_rank(target, tau)
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    corr = (rp * rt).sum() / (torch.sqrt((rp ** 2).sum()) * torch.sqrt((rt ** 2).sum()) + 1e-8)
    return 1.0 - corr


def feature_mask_branches(feat, K):
    # BFT-D: K deterministic feature-masked views of one feature batch.
    # Branch k zeroes the k-th contiguous 1/K slice of the feature vector.
    B, D = feat.shape
    branches = []
    for k in range(K):
        masked = feat.clone()
        start, end = int(k / K * D), int((k + 1) / K * D)
        masked[:, start:end] = 0.0
        branches.append(masked)
    return branches


def freq_shift(x, f_shift, sample_rate):
    # Hilbert-transform frequency shift along the time axis, length preserving.
    # x: [B, 1, C, T] tensor. Returns a shifted [B, 1, C, T] tensor.
    device = x.device
    arr = x.detach().cpu().numpy()
    B, _, C, T = arr.shape
    n = 1
    while n < T:
        n *= 2
    t = np.arange(n)
    shift_func = np.exp(2j * np.pi * f_shift * (1.0 / sample_rate) * t)
    out = np.zeros_like(arr)
    for b in range(B):
        for c in range(C):
            padded = np.zeros(n)
            padded[:T] = arr[b, 0, c, :]
            out[b, 0, c, :] = (hilbert(padded) * shift_func)[:T].real
    return torch.tensor(out, dtype=torch.float32, device=device)


def augment_branches(x, args):
    # BFT-A: K knowledge-guided input augmentations of one trial batch.
    # x: [B, 1, C, T] tensor. Returns a list of K [B, 1, C, T] tensors.
    branches = [x]                                             # identity
    branches.append(x + (torch.rand_like(x) - 0.5) * x.std() / 2.0)  # Gaussian noise
    for m in (0.1, -0.1, -0.2):                               # multiplicative scaling
        branches.append(x * (1 - m))
    branches.append(freq_shift(x, 0.2, args.sample_rate))     # frequency shift up
    branches.append(freq_shift(x, -0.2, args.sample_rate))    # frequency shift down
    step = max(1, int(0.2 * args.sample_rate))                # temporal shifts
    for no in (1, 2, 3, 4, 5):
        branches.append(torch.roll(x, shifts=step * no, dims=-1))
    return branches                                           # K = 12


def train_ranker(netF, netC, ranker, loader, args):
    # Train r(.) on source data so that its per-view reliability ranking matches
    # the real per-view task-loss ranking. Only r(.) is updated; the backbone
    # stays frozen (backpropagation-free with respect to the source model).
    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ranker.parameters(), lr=args.ranker_lr)
    netF.eval()
    netC.eval()
    ranker.train()

    max_iter = args.ranker_epoch * len(loader)
    iter_num = 0
    iter_source = iter(loader)
    while iter_num < max_iter:
        try:
            inputs, labels = next(iter_source)
        except StopIteration:
            iter_source = iter(loader)
            inputs, labels = next(iter_source)
        if inputs.size(0) <= 1:
            continue
        iter_num += 1
        if args.data_env != 'local':
            inputs, labels = inputs.cuda(), labels.cuda()

        real_losses, pred_losses = [], []
        with torch.no_grad():
            base_feat = netF(inputs)
        if args.variant == 'BFT-A':
            views = augment_branches(inputs, args)
            for v in views:
                with torch.no_grad():
                    feat = netF(v)
                    _, logits = netC(feat)
                    real_losses.append(ce(logits, labels).item())
                pred_losses.append(ranker(feat).mean())
        else:
            for masked in feature_mask_branches(base_feat, args.K):
                with torch.no_grad():
                    _, logits = netC(masked)
                    real_losses.append(ce(logits, labels).item())
                pred_losses.append(ranker(masked).mean())

        real_losses = torch.tensor(real_losses)
        if args.data_env != 'local':
            real_losses = real_losses.cuda()
        target = F.softmax(-real_losses, dim=0)
        pred = F.softmax(-torch.stack(pred_losses).squeeze(), dim=0)

        loss = soft_spearman_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ranker.eval()


def BFT(X_tar, y_tar, netF, netC, ranker, args):
    # Online, backpropagation-free test-time adaptation over the target stream.
    # Per trial: incremental Euclidean Alignment, build K views, weight them by
    # the running-averaged ranker reliability, and aggregate the softmax outputs.
    # The feature extractor's batch-norm statistics are refreshed on a sliding
    # window of recent trials, which is the only test-time state change.
    ranker.eval()
    T = args.temperature
    all_output, all_label, all_probs = [], [], None
    R = 0
    n = X_tar.shape[0]
    for i in range(n):
        netF.eval()
        netC.eval()
        sample = X_tar[i].reshape(args.chn, args.time_sample_num)
        if args.align:
            R = EA_online(sample, R, i)
            sample = np.dot(fractional_matrix_power(R, -0.5), sample)
        x = sample.reshape(1, 1, args.chn, args.time_sample_num)
        x = torch.from_numpy(x).to(torch.float32)
        if args.data_env != 'local':
            x = x.cuda()

        if i == 0:
            data_cum = x.detach().clone()
        else:
            data_cum = torch.cat((data_cum, x.detach().clone()), 0)

        with torch.no_grad():
            # per-view logits and reliability scores
            view_logits, pred_losses = [], []
            if args.variant == 'BFT-A':
                for v in augment_branches(x, args):
                    feat = netF(v)
                    _, logits = netC(feat)
                    view_logits.append(logits)
                    pred_losses.append(ranker(feat))
            else:
                feat = netF(x)
                for masked in feature_mask_branches(feat, args.K):
                    _, logits = netC(masked)
                    view_logits.append(logits)
                    pred_losses.append(ranker(masked))

            pred_losses = torch.stack(pred_losses).squeeze()
            weights = F.softmax(-pred_losses, dim=0)
            # running average of the per-view reliability weights
            all_probs = weights.unsqueeze(0) if all_probs is None \
                else torch.cat((all_probs, weights.unsqueeze(0)), 0)
            probs = all_probs.mean(dim=0)

            views = torch.stack([nn.Softmax(dim=1)(l / T) for l in view_logits]).squeeze(1)
            mean_output = (views * probs.unsqueeze(1)).sum(dim=0) / probs.sum()
            all_output.append(mean_output.float().cpu().numpy())
            all_label.append(y_tar[i])

        # refresh BN running statistics on a sliding window (no gradients)
        if (i + 1) >= args.test_batch:
            netF.train()
            batch = data_cum[i - args.test_batch + 1:i + 1]
            batch = batch.reshape(args.test_batch, 1, args.chn, args.time_sample_num)
            with torch.no_grad():
                _ = netF(batch)
            netF.eval()

    pred = np.argmax(np.array(all_output).reshape(-1, args.class_num), axis=1)
    return accuracy_score(all_label, pred) * 100


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    netF, netC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

    # load the frozen source model (same source-combined EA+EEGNet baseline as
    # the other methods under ./runs); train.sh / dnn.py produce these.
    ckpt = './runs/' + str(args.data_name) + '/' + str(args.backbone) + \
        '_S' + str(args.idt) + '_seed' + str(args.SEED) + '.ckpt'
    if args.data_env != 'local':
        base_network.load_state_dict(torch.load(ckpt))
    else:
        base_network.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))
    base_network.eval()

    # train the reliability ranker once on source data, then run BFT on target
    ranker = ReliabilityRanker(args.feature_deep_dim)
    if args.data_env != 'local':
        ranker = ranker.cuda()
    train_ranker(netF, netC, ranker, dset_loaders["source"], args)

    acc = BFT(X_tar, y_tar, netF, netC, ranker, args)
    log_str = 'Task: {}, {} Test Acc = {:.2f}%'.format(args.task_str, args.variant, acc)
    args.log.record(log_str)
    print(log_str)

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()
    return acc


if __name__ == '__main__':

    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']

    dct = pd.DataFrame(columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4',
                                's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13'])

    for data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640

        # BFT variant: 'BFT-D' (feature-masked subnetworks) or 'BFT-A' (input augmentations)
        variant = 'BFT-D'

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, trial_num=trial_num,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, paradigm=paradigm,
                                  data_name=data_name, variant=variant)

        args.method = 'BFT'
        args.backbone = 'EEGNet'
        # whether to use Euclidean Alignment
        args.align = True
        # number of feature-masked branches for BFT-D (BFT-A uses a fixed bank of 12)
        args.K = 10
        # sliding window of recent trials used to refresh BN statistics
        args.test_batch = 8
        # aggregation temperature and ranker training schedule
        args.temperature = 0.25
        args.ranker_lr = 0.001
        args.ranker_epoch = 20
        args.batch_size = 32

        # GPU device id
        try:
            device_id = str(sys.argv[1])
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
        except:
            args.data_env = 'local'
        total_acc = []

        for s in [1, 2, 3]:
            args.SEED = s
            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True

            args.data = data_name
            print(args.data)
            print(args.method)
            print(args.SEED)
            print(args)

            args.local_dir = './data/' + str(data_name) + '/'
            args.result_dir = './logs/'
            my_log = LogRecord(args)
            my_log.log_init()
            my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

            sub_acc_all = np.zeros(N)
            for idt in range(N):
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                print(info_str)
                my_log.record(info_str)
                args.log = my_log

                sub_acc_all[idt] = train_target(args)
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            total_acc.append(sub_acc_all)

            args.log.record("\n==========================================")
            args.log.record(str(np.round(sub_acc_all, 3).tolist()))
            args.log.record(str(np.round(np.mean(sub_acc_all), 3).tolist()))

        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)
        print(str(total_acc))
        args.log.record(str(total_acc))

        subject_mean = np.round(np.average(total_acc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_acc)), 5)
        total_std = np.round(np.std(np.average(total_acc, axis=1)), 5)
        print(subject_mean)
        print(total_mean)
        print(total_std)

        result_dct = {'dataset': data_name, 'avg': total_mean, 'std': total_std}
        for i in range(len(subject_mean)):
            result_dct['s' + str(i)] = subject_mean[i]
        dct = dct.append(result_dct, ignore_index=True)

    dct.to_csv('./logs/' + str(args.method) + ".csv")
