# -*- coding: utf-8 -*-
# @Time    : 2023/07/13
# @Author  : Siyang Li
# @File    : domain_classifier_pretest.py
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils.network import backbone_net
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_split
from utils.utils import fix_random_seed, cal_acc_comb, data_loader_split

import gc
import sys


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_mi_combine_split(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    dset_loaders = data_loader_split(X_src, y_src, X_tar, y_tar, args)

    netF, netC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

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

            acc_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network, args=args)
            log_str = 'Iter:{}/{}; Test Acc = {:.2f}%'.format(int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
            args.log.record(log_str)
            print(log_str)

            base_network.train()

    print('Test Acc = {:.2f}%'.format(acc_t_te))

    print('saving model...')

    if args.align:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_seed' + str(args.SEED) + '.ckpt')
    else:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_seed' + str(args.SEED) + '_noEA' + '.ckpt')

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()

    return acc_t_te


if __name__ == '__main__':

    # use all domains, split in half, check if domain classifier can really work

    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014001-4']

    for data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 9, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 14, 2561, 512, 100, 640
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 12, 2561, 512, 200, 640
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 9, 1001, 250, 288, 248

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, trial_num=trial_num,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=data_name)

        args.method = 'Domain_Classifier_pretest'
        args.backbone = 'EEGNet'

        # whether to use EA
        args.align = True

        # train test split ratio
        args.ratio = 0.5

        # learning rate
        args.lr = 0.001

        # train batch size
        args.batch_size = 32

        # training epochs
        args.max_epoch = 100

        # GPU device id
        try:
            device_id = str(sys.argv[1])
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
        except:
            args.data_env = 'local'

        total_acc = []

        # train multiple randomly initialized models
        for s in [1, 2, 3, 4, 5]:
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

            info_str = '\n==========================Domain Classifier=========================='
            print(info_str)
            my_log.record(info_str)
            args.log = my_log

            all_acc = train_target(args)
            print('Avg acc: ', np.round(all_acc, 3))
            total_acc.append(all_acc)

            args.log.record("\n==========================================")

        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)

        print(str(total_acc))

        args.log.record(str(total_acc))

        total_mean = np.round(np.average(total_acc), 5)
        total_std = np.round(np.std(total_acc), 5)

        print(total_mean)
        print(total_std)

        args.log.record(str(total_mean))
        args.log.record(str(total_std))
