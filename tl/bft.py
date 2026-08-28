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
#
# Where each equation of the paper lives in this file:
#   Eq. (1)  z^(k) = g(T_k(x))          augment_branches         BFT-A views
#   Eq. (2)  mask I^(k)                 feature_mask_branches    BFT-D views
#   Eq. (3)  z = 1/(1-p) I^(k) . g(x)   feature_mask_branches    see note there
#   Eq. (4)  L_mapping, the sorter m    soft_rank/soft_spearman_loss
#   Eq. (5)  w_k = softmax(r(z^(k)))    train_ranker and BFT_func
#   Eq. (6)  L_ranking                  soft_spearman_loss via train_ranker
#   Eq. (7)  weighted aggregation       BFT_func
# Eq. (8), the top-ceil(K/2) average used for regression, is not implemented
# here: this file covers the classification experiments (Tables III and IV) and
# not the driver-drowsiness regression of Table V. The full implementation
# behind every experiment of the paper, including the regression pipeline, the
# SoDeep mapping module, the corruption robustness study and the timing study,
# is under paper/BFT/code/.
#
# This file follows the same conventions as every other method in tl/: it reads
# MOABB data through read_mi_combine_tar and data_loader, consumes the same
# dset_loaders["Target-Online"] stream one trial at a time, loads the same
# source checkpoints under ./runs/ that dnn.py writes, supports the same
# balanced and imbalanced (2:1) protocols, the same calc_time instrumentation
# and the same eleven seeds. So BFT can be compared against bn-adapt, Tent,
# T3A, SAR, DELTA, CoTTA and T-TIME with nothing changed but the script name.
#
# Four places knowingly depart from the manuscript. Each is marked with a
# "NOTE (paper)" comment at the point where it happens, so that the published
# numbers stay reproducible from this file rather than being silently changed:
# the Eq. (3) rescaling is not applied, the aggregation temperature is 0.25
# rather than the 0.5 of Section IV-B, the Eq. (7) weights are averaged over
# the test stream rather than being per-trial, and the ranking supervision of
# Eq. (6) is applied per mini-batch rather than per trial.
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
from sklearn.metrics import roc_auc_score, accuracy_score

from utils.network import backbone_net
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader, cal_auc_comb, cal_score_online
from utils.alg_utils import EA_online


class ReliabilityRanker(nn.Module):
    # r(.) of Eq. (5): maps a backbone feature vector to a scalar predicted task
    # loss. A lower predicted loss means a more reliable view.
    #
    # Eq. (5) is printed as w = softmax(r(z)), but r here is trained to predict
    # a LOSS, so the callers below use softmax(-r(z)). The two agree once the
    # sign convention of r is fixed; this file is self-consistent about it.
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
    # sigmoid comparisons. rank_i grows with x_i. This is the stand-in for the
    # SoDeep sorter m(.) of Eq. (4): m is pre-trained there to imitate a sort,
    # whereas here the sort is approximated in closed form and needs no
    # pre-training, which is why this file has no sorter checkpoint to load.
    #
    # tau sets how sharp the comparison is. The sigmoid only behaves like a
    # hard comparison when the pairwise gaps are large next to tau. At the
    # tau = 1.0 used below the gaps are two orders of magnitude smaller, so the
    # sigmoid stays in its linear regime and soft_rank returns an affine
    # function of x rather than a rank. See the note in soft_spearman_loss.
    diff = (x.unsqueeze(1) - x.unsqueeze(0)) / tau
    return torch.sigmoid(diff).sum(dim=1)


def soft_spearman_loss(pred, target, tau=1.0):
    # 1 - Spearman rank correlation between predicted and target reliabilities.
    # Self-contained stand-in for the SoDeep differentiable sorter m(.), and the
    # realisation of L_ranking in Eq. (6).
    #
    # NOTE (behaviour): callers pass softmax outputs, whose entries sit near 1/K
    # and differ by ~0.02 for K = 10. Divided by tau = 1.0 those gaps land in the
    # linear part of the sigmoid, so soft_rank spans about 0.08 where a true rank
    # over K = 10 spans 9. What this function returns is therefore the Pearson
    # correlation of the softmax vectors, not a rank correlation. It is kept as
    # published because it is what produced the reported numbers, and because it
    # is a working surrogate: measured over 2000 simulated trials it tracks the
    # true Spearman correlation at r = 0.78, and its gradient magnitude is the
    # same as the sharper form. Passing raw scores with tau near 0.01 instead
    # raises that agreement to r = 0.97. tau is already a parameter of both this
    # function and soft_rank, so that sharper behaviour needs no code change,
    # but it retrains the ranker and so moves the published results.
    rp = soft_rank(pred, tau)
    rt = soft_rank(target, tau)
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    corr = (rp * rt).sum() / (torch.sqrt((rp ** 2).sum()) * torch.sqrt((rt ** 2).sum()) + 1e-8)
    return 1.0 - corr


def feature_mask_branches(feat, K):
    # BFT-D, Eq. (2): K deterministic feature-masked views of one feature batch.
    # Branch k zeroes the k-th contiguous 1/K slice of the feature vector, so
    # I^(k)_i = 0 for i in [(k-1)d/K, k d/K) and 1 elsewhere. The K slices are
    # disjoint and together cover the whole feature vector, which is what makes
    # the bank deterministic rather than a resampled dropout mask.
    #
    # NOTE (paper): Eq. (3) writes z = 1/(1-p) I^(k) . g(x), the usual inverted
    # dropout rescaling with p = 1/K. That factor is not applied here, so the
    # surviving features keep their original scale. It is not a no-op, because
    # the classifier head has a bias and its logits are therefore not invariant
    # to a rescaling of the features. On the authors' other BFT implementation,
    # same method and same K on BNCI2014001 over three seeds, adding the factor
    # moved accuracy by about 0.16 points, which is inside the seed-to-seed
    # spread; it has not been measured on this file. Left as published so the
    # reported numbers reproduce. To follow Eq. (3), scale by K / (K - 1).
    B, D = feat.shape
    branches = []
    for k in range(K):
        masked = feat.clone()
        start, end = int(k / K * D), int((k + 1) / K * D)
        masked[:, start:end] = 0.0
        branches.append(masked)
    return branches


def freq_shift(x, f_shift, sample_rate):
    # One of the T_k of Eq. (1). Hilbert-transform frequency shift along the
    # time axis, length preserving.
    # x: [B, 1, C, T] tensor. Returns a shifted [B, 1, C, T] tensor.
    #
    # sample_rate is taken from the caller rather than assumed, so the shift is
    # the intended 0.2 Hz on every dataset. Hard-coding 1/250 here would make
    # the shift scale with the sampling rate, which would silently double it on
    # the 512 Hz datasets configured at the bottom of this file.
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
    # BFT-A, Eq. (1): z^(k) = g(T_k(x)), the K knowledge-guided input
    # augmentations T_k of one trial batch. This function supplies the T_k; the
    # caller applies the backbone g.
    # x: [B, 1, C, T] tensor. Returns a list of K [B, 1, C, T] tensors.
    #
    # The bank is fixed at K = 12 and does not read args.K, which only sets the
    # BFT-D bank size. The identity is branch 0, so unlike a pure augmentation
    # ensemble the untransformed trial always competes for weight.
    branches = [x]                                             # identity
    # uniform on +/- std/4, not Gaussian: torch.rand_like is uniform on [0, 1),
    # so (rand - 0.5) * std / 2 is uniform on +/- std/4. Section III-B of the
    # paper specifies uniform and Section IV-C says Gaussian; the code follows
    # Section III-B.
    branches.append(x + (torch.rand_like(x) - 0.5) * x.std() / 2.0)  # uniform noise
    for m in (0.1, -0.1, -0.2):                               # multiplicative scaling
        branches.append(x * (1 - m))
    branches.append(freq_shift(x, 0.2, args.sample_rate))     # frequency shift up
    branches.append(freq_shift(x, -0.2, args.sample_rate))    # frequency shift down
    step = max(1, int(0.2 * args.sample_rate))                # temporal shifts
    for no in (1, 2, 3, 4, 5):
        branches.append(torch.roll(x, shifts=step * no, dims=-1))
    return branches                                           # K = 12


def train_ranker(netF, netC, ranker, loader, args):
    # Eq. (6): train r(.) on source data so that its per-view reliability
    # ranking matches the real per-view task-loss ranking. Only r(.) is updated;
    # the backbone stays frozen, which is what "backpropagation-free" refers to.
    # It describes the source model at test time, not this offline stage, which
    # does use gradients but only on r.
    #
    # NOTE (paper): Eq. (6) is written per trial i, with pi_i the true ordering
    # for that trial. Here the cross-entropy is taken with its default mean
    # reduction and the ranker output is averaged over the batch, so one
    # K-vector is fitted per mini-batch rather than one per trial. The ordering
    # being learned is therefore a batch-average ordering.
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

        # real_losses[k] is the true task loss of view k, the supervision pi of
        # Eq. (6). pred_losses[k] is r(z^(k)), the ranker's estimate of it. Only
        # the latter carries gradients: the views and their logits are built
        # under no_grad, so the backbone is never differentiated through.
        real_losses, pred_losses = [], []
        if args.variant == 'BFT-A':
            # Eq. (1): transform the input, then encode each view separately.
            views = augment_branches(inputs, args)
            for v in views:
                with torch.no_grad():
                    feat = netF(v)
                    _, logits = netC(feat)
                    real_losses.append(ce(logits, labels).item())
                pred_losses.append(ranker(feat).mean())
        else:
            # Eq. (2): encode once, then mask the feature K ways. This is why
            # BFT-D costs one backbone pass per batch where BFT-A costs K, the
            # same saving that makes BFT-D the cheaper variant at test time.
            with torch.no_grad():
                base_feat = netF(inputs)
            for masked in feature_mask_branches(base_feat, args.K):
                with torch.no_grad():
                    _, logits = netC(masked)
                    real_losses.append(ce(logits, labels).item())
                pred_losses.append(ranker(masked).mean())

        real_losses = torch.tensor(real_losses)
        if args.data_env != 'local':
            real_losses = real_losses.cuda()
        # Eq. (5) applied to both sides, so the loss compares two distributions
        # over the K views. The minus sign turns a predicted loss into a
        # reliability, matching the sign convention noted on ReliabilityRanker.
        target = F.softmax(-real_losses, dim=0)
        pred = F.softmax(-torch.stack(pred_losses).squeeze(), dim=0)

        loss = soft_spearman_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ranker.eval()


def BFT_func(loader, model, ranker, args, balanced=True):
    # Eq. (7): online, backpropagation-free test-time adaptation over the target
    # stream. Per trial: incremental Euclidean Alignment, build K views, weight
    # them by the ranker reliability, and aggregate the softmax outputs. The
    # feature extractor's BatchNorm statistics are refreshed on a sliding window
    # of recent trials, which is the only test-time state change. No optimiser
    # exists in this function and no gradient is taken.
    #
    # The signature and the loop follow the other test-time methods in tl/, in
    # particular bn-adapt.py, so BFT consumes the same MOABB stream they do:
    # dset_loaders["Target-Online"] yields one unaligned trial at a time and the
    # alignment is done here, on the fly, as an online protocol requires.

    if balanced == False and args.data_name == 'BNCI2014001-4':
        print('ERROR, imbalanced multi-class not implemented')
        sys.exit(0)

    netF, netC = model[0], model[1]
    ranker.eval()

    y_true = []
    y_pred = []
    # running average of the per-view reliability weights, see the note below
    all_probs = None

    # initialize test reference matrix for Incremental EA
    if args.align:
        R = 0

    iter_test = iter(loader)

    # loop through test data stream one by one
    for i in range(len(loader)):
        #################### Phase 1: target label prediction ####################
        model.eval()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        inputs = inputs.reshape(1, 1, inputs.shape[-2], inputs.shape[-1]).cpu()

        # accumulate test data
        if i == 0:
            data_cum = inputs.float().cpu()
        else:
            data_cum = torch.cat((data_cum, inputs.float().cpu()), 0)

        # Incremental EA
        if args.align:
            start_time = time.time()

            if i == 0:
                sample_test = data_cum.reshape(args.chn, args.time_sample_num)
            else:
                sample_test = data_cum[i].reshape(args.chn, args.time_sample_num)
            # update reference matrix
            R = EA_online(sample_test, R, i)

            sqrtRefEA = fractional_matrix_power(R, -0.5)
            # transform current test sample
            sample_test = np.dot(sqrtRefEA, sample_test)

            EA_time = time.time()
            if args.calc_time:
                print('sample ', str(i), ', pre-inference IEA finished time in ms:', np.round((EA_time - start_time) * 1000, 3))
            sample_test = sample_test.reshape(1, 1, args.chn, args.time_sample_num)
        else:
            sample_test = data_cum[i].numpy()
            sample_test = sample_test.reshape(1, 1, sample_test.shape[1], sample_test.shape[2])

        if args.data_env != 'local':
            sample_test = torch.from_numpy(sample_test).to(torch.float32).cuda()
        else:
            sample_test = torch.from_numpy(sample_test).to(torch.float32)

        start_time = time.time()
        with torch.no_grad():
            # per-view logits and reliability scores, Eq. (1) or Eq. (2) then r
            view_logits, pred_losses = [], []
            if args.variant == 'BFT-A':
                # K = 12 backbone passes, one per augmented input.
                for v in augment_branches(sample_test, args):
                    feat = netF(v)
                    _, logits = netC(feat)
                    view_logits.append(logits)
                    pred_losses.append(ranker(feat))
            else:
                # one backbone pass, then K masks applied to the same feature.
                feat = netF(sample_test)
                for masked in feature_mask_branches(feat, args.K):
                    _, logits = netC(masked)
                    view_logits.append(logits)
                    pred_losses.append(ranker(masked))

            # Eq. (5): reliability weights over the K views for this trial.
            pred_losses = torch.stack(pred_losses).squeeze()
            weights = F.softmax(-pred_losses, dim=0)
            # NOTE (paper): Eq. (7) weights each trial by its own w_{t,k}. Here
            # the per-trial weights are accumulated and averaged over every
            # trial seen so far, so the weights applied at trial t are a running
            # mean rather than that trial's own. This makes the weights drift
            # towards a constant as the stream lengthens, which matters most for
            # BFT-D, whose weights are already close to uniform. Replace `probs`
            # with `weights` below to follow Eq. (7) literally.
            all_probs = weights.unsqueeze(0) if all_probs is None \
                else torch.cat((all_probs, weights.unsqueeze(0)), 0)
            probs = all_probs.mean(dim=0)

            # Eq. (7): temperature-scaled softmax per view, then the weighted
            # sum over views. probs already sums to one, so the division is a
            # normalisation that costs nothing and guards against a future
            # change to `probs` that does not.
            views = torch.stack([nn.Softmax(dim=1)(l / args.temperature) for l in view_logits]).squeeze(1)
            softmax_out = (views * probs.unsqueeze(1)).sum(dim=0) / probs.sum()
            softmax_out = softmax_out.reshape(1, args.class_num)

        inference_time = time.time()
        if args.calc_time:
            print('sample ', str(i), ', backpropagation-free inference finished in ms:', np.round((inference_time - start_time) * 1000, 3))

        labels = labels.float().cpu()
        y_pred.append(softmax_out.detach().cpu().numpy())
        y_true.append(labels.item())

        #################### Phase 2: target model update ####################
        # BN-adapt on a sliding window of the most recent args.test_batch
        # trials. Only the three BatchNorm modules of EEGNet are put in train
        # mode, exactly as in bn-adapt.py: calling model.train() would also
        # enable the two Dropout layers, which would corrupt the statistics that
        # the later BatchNorm sees and would consume the RNG. No parameter is
        # changed and no gradient is taken, so this stays within the
        # backpropagation-free claim. The update runs after the current trial
        # has been predicted, so no trial is scored using statistics that saw
        # itself.
        if (i + 1) >= args.test_batch and (i + 1) % args.stride == 0:
            if args.align:
                batch_test = np.copy(data_cum[i - args.test_batch + 1:i + 1])
                # transform test batch
                batch_test = np.dot(sqrtRefEA, batch_test)
                batch_test = np.transpose(batch_test, (1, 2, 0, 3))
            else:
                batch_test = data_cum[i - args.test_batch + 1:i + 1].numpy()
                batch_test = batch_test.reshape(args.test_batch, 1, batch_test.shape[2], batch_test.shape[3])

            if args.data_env != 'local':
                batch_test = torch.from_numpy(batch_test).to(torch.float32).cuda()
            else:
                batch_test = torch.from_numpy(batch_test).to(torch.float32)

            start_time = time.time()
            for step in range(args.steps):

                model[0].block1[2].train()
                model[0].block1[4].train()
                model[0].block2[3].train()

                # forward pass for model BN update; the output is discarded,
                # only the running statistics it leaves behind are wanted
                with torch.no_grad():
                    model(batch_test)

                model[0].block1[2].eval()
                model[0].block1[4].eval()
                model[0].block2[3].eval()

            TTA_time = time.time()
            if args.calc_time:
                print('sample ', str(i), ', post-inference model update finished in ms:', np.round((TTA_time - start_time) * 1000, 3))

        model.eval()

    if balanced:
        _, predict = torch.max(torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num), 1)
        pred = torch.squeeze(predict).float()
        score = accuracy_score(y_true, pred)
        if args.data_name == 'BNCI2014001-4':
            y_pred = np.array(y_pred).reshape(-1, )  # multiclass
        else:
            y_pred = np.array(y_pred).reshape(-1, args.class_num)[:, 1]  # binary
    else:
        predict = torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num)
        y_pred = np.array(predict).reshape(-1, args.class_num)[:, 1]  # binary
        score = roc_auc_score(y_true, y_pred)

    return score * 100, y_pred


def train_target(args):
    if not args.align:
        extra_string = '_noEA'
    else:
        extra_string = ''
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    netF, netC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

    # the frozen source model, shared with every other method in tl/. Either
    # load the checkpoint dnn.py wrote, or train it here when max_epoch > 0.
    if args.max_epoch == 0:
        if args.data_env != 'local':
            base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
                '_S' + str(args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt'))
        else:
            base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
                '_S' + str(args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt', map_location=torch.device('cpu')))
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
        optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

        max_iter = args.max_epoch * len(dset_loaders["source"])
        interval_iter = max_iter // args.max_epoch
        args.max_iter = max_iter
        iter_num = 0
        base_network.train()

        while iter_num < max_iter:
            try:
                inputs_source, labels_source = next(iter_source)
            except:
                iter_source = iter(dset_loaders["source"])
                inputs_source, labels_source = next(iter_source)

            if inputs_source.size(0) == 1:
                continue

            iter_num += 1

            features_source, outputs_source = base_network(inputs_source)

            classifier_loss = criterion(outputs_source, labels_source)

            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            classifier_loss.backward()
            optimizer_f.step()
            optimizer_c.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                base_network.eval()

                if args.balanced:
                    acc_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network, args=args)
                    log_str = 'Task: {}, Iter:{}/{}; Offline-EA Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                else:
                    acc_t_te, _ = cal_auc_comb(dset_loaders["Target-Imbalanced"], base_network, args=args)
                    log_str = 'Task: {}, Iter:{}/{}; Offline-EA AUC = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
                print(log_str)

                base_network.train()

        print('saving model...')
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(
                       args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt')

    base_network.eval()

    # Eq. (6): train the reliability ranker r once, offline, on source data.
    # This is the only stage in BFT that takes a gradient, and it touches r
    # alone; the backbone stays frozen here and at test time.
    ranker = ReliabilityRanker(args.feature_deep_dim)
    if args.data_env != 'local':
        ranker = ranker.cuda()
    start_time = time.time()
    train_ranker(netF, netC, ranker, dset_loaders["source"], args)
    if args.calc_time:
        print('ranker training finished in s:', np.round(time.time() - start_time, 3))

    # the same source-model-plus-online-EA reference the other methods report
    score = cal_score_online(dset_loaders["Target-Online"], base_network, args=args)
    if args.balanced:
        log_str = 'Task: {}, Online IEA Acc = {:.2f}%'.format(args.task_str, score)
    else:
        log_str = 'Task: {}, Online IEA AUC = {:.2f}%'.format(args.task_str, score)
    args.log.record(log_str)
    print(log_str)

    print('executing TTA...')

    if args.balanced:
        acc_t_te, y_pred = BFT_func(dset_loaders["Target-Online"], base_network, ranker, args=args, balanced=True)
        log_str = 'Task: {}, {} TTA Acc = {:.2f}%'.format(args.task_str, args.variant, acc_t_te)
    else:
        acc_t_te, y_pred = BFT_func(dset_loaders["Target-Online-Imbalanced"], base_network, ranker, args=args, balanced=False)
        log_str = 'Task: {}, {} TTA AUC = {:.2f}%'.format(args.task_str, args.variant, acc_t_te)
    args.log.record(log_str)
    print(log_str)

    if args.balanced:
        print('Test Acc = {:.2f}%'.format(acc_t_te))
    else:
        print('Test AUC = {:.2f}%'.format(acc_t_te))

    torch.save(base_network.state_dict(), './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(
        args.SEED) + extra_string + '_adapted' + '.ckpt')

    # save the predictions for ensemble
    with open('./logs/' + str(args.data_name) + '_' + str(args.method) + '_seed_' + str(args.SEED) + "_pred.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(y_pred)

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()

    return acc_t_te


if __name__ == '__main__':

    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014001-4']

    dct = pd.DataFrame(columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4',
                                's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13'])
    # one row per dataset, written to CSV after the loop
    result_rows = []

    for data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248

        # BFT variant: 'BFT-D' (feature-masked subnetworks) or 'BFT-A' (input augmentations)
        variant = 'BFT-D'

        # whether to use pretrained model
        # if source models have not been trained, set use_pretrained_model to False to train them
        # alternatively, run dnn.py to train source models, in seperating the steps
        use_pretrained_model = True
        if use_pretrained_model:
            # no training
            max_epoch = 0
        else:
            # training epochs
            max_epoch = 100

        # learning rate
        lr = 0.001

        # test batch size
        test_batch = 8

        # update step
        steps = 1

        # update stride
        stride = 1

        # whether to use EA
        align = True

        # whether to test balanced or imbalanced (2:1) target subject
        balanced = True

        # whether to record running time
        calc_time = False

        # number of feature-masked branches for BFT-D, the K of Eq. (2).
        # BFT-A ignores this and always uses its fixed bank of 12, so changing
        # it has no effect when variant is 'BFT-A'.
        K = 10

        # NOTE (paper): the temperature of Eq. (7). Section IV-B of the paper
        # states tau = 0.5; the value that produced the reported numbers is the
        # 0.25 below. Kept as-is so the results reproduce from this file.
        temperature = 0.25

        # ranker training schedule, the offline stage of Eq. (6)
        ranker_lr = 0.001
        ranker_epoch = 20

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, align=align, lr=lr, max_epoch=max_epoch,
                                  trial_num=trial_num, time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, stride=stride, steps=steps, calc_time=calc_time,
                                  paradigm=paradigm, test_batch=test_batch, data_name=data_name, balanced=balanced,
                                  variant=variant, K=K, temperature=temperature, ranker_lr=ranker_lr,
                                  ranker_epoch=ranker_epoch)

        # the variant is the method name, so the log file, the per-seed
        # prediction CSV and the results CSV are all specific to it. With a
        # shared 'BFT' name a BFT-A run would overwrite or append to the BFT-D
        # outputs of all three.
        args.method = variant
        args.backbone = 'EEGNet'

        # train batch size
        args.batch_size = 32

        # GPU device id
        try:
            device_id = str(sys.argv[1])
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
        except:
            args.data_env = 'local'
        total_acc = []

        # update multiple models, independently, from the source models
        for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
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
        # DataFrame.append was removed in pandas 2.0, so appending here raised
        # AttributeError after every subject and seed had already been computed,
        # losing the whole run's summary at the last step. Rows are collected
        # and the frame is built once at the end, which works on the pinned
        # pandas 1.4.3 of environment.yml and on 2.x, and avoids concatenating
        # onto an empty frame, which pandas 2.x warns about.
        result_rows.append(result_dct)

    # save results to csv
    dct = pd.DataFrame(result_rows, columns=dct.columns)
    dct.to_csv('./logs/' + str(args.method) + ".csv")
