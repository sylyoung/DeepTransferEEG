# -*- coding: utf-8 -*-
# @Time    : 2023/01/11
# @Author  : Siyang Li
# @File    : shot.py
import csv

import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from utils.network import backbone_net
from utils.loss import Entropy
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader
from utils.utils import lr_scheduler, fix_random_seed, op_copy, cal_acc, cal_bca, cal_auc

import gc
import torch
import sys


def obtain_label(loader, netF, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netF(inputs)
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return predict.astype('int')


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    netF, netC = backbone_net(args, return_type='y')
    if args.data_env != 'local':
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

    criterion = nn.CrossEntropyLoss()

    ######################################################################################################
    # Source Model Training
    ######################################################################################################

    optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

    args.batch_size = 32
    # TODO load pretrained model
    args.max_epoch = 0

    if args.max_epoch == 0:
        base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
                                                '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_EA_' + str(args.align) + '.ckpt'))
        #base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
        #                                        '_S' + str(args.idt) + '_seed' + str(args.SEED)+ + '_EA_' + str(args.align) + '_Imbalanced' + '.ckpt'))
    else:
        max_iter = args.max_epoch * len(dset_loaders["source"])
        #max_iter = args.max_epoch * len(dset_loaders["source-Imbalanced"])
        interval_iter = max_iter // args.max_epoch
        args.max_iter = max_iter
        iter_num = 0
        base_network.train()

        print('Source Model Training')

        while iter_num < max_iter:
            try:
                inputs_source, labels_source = next(iter_source)
            except:
                iter_source = iter(dset_loaders["source"])
                #iter_source = iter(dset_loaders["source-Imbalanced"])
                inputs_source, labels_source = next(iter_source)

            if inputs_source.size(0) == 1:
                continue

            iter_num += 1

            outputs_source = base_network(inputs_source)

            outputs_source = torch.nn.Softmax(dim=1)(outputs_source / 2)

            args.trade_off = 1.0
            classifier_loss = criterion(outputs_source, labels_source)
            total_loss = classifier_loss

            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            total_loss.backward()
            optimizer_f.step()
            optimizer_c.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                base_network.eval()

                acc_t_te, _ = cal_acc(dset_loaders["Target"], netF, netC, args=args)
                log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                #log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source-Imbalanced"])), int(max_iter // len(dset_loaders["source-Imbalanced"])),acc_t_te)
                args.log.record(log_str)
                print(log_str)

                base_network.train()

    ######################################################################################################
    # Source HypOthesis Transfer
    ######################################################################################################

    print('Source HypOthesis Transfer')

    args.batch_size = 32
    args.max_epoch = 5
    #dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    netC.eval()
    netF.train()

    '''
    # SHOT-original, commented out for SHOT-IM
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False

    optimizer = optim.Adam(param_group)
    optimizer = op_copy(optimizer)
    '''
    optimizer = optim.Adam(netF.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    #max_iter = args.max_epoch * len(dset_loaders["target-Imbalanced"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, _ = next(iter_test)
            tar_id += 1
            tar_idx = np.arange(args.batch_size, dtype=int) + args.batch_size * tar_id
        except:
            iter_test = iter(dset_loaders["target"])
            #iter_test = iter(dset_loaders["target-Imbalanced"])
            inputs_test, _ = next(iter_test)
            tar_id = 0
            tar_idx = np.arange(args.batch_size, dtype=int)

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            mem_label = obtain_label(dset_loaders["Target"], netF, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()

        iter_num += 1
        #lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        features_test = netF(inputs_test)
        outputs_test = netC(features_test)

        # loss definition
        if args.cls_par > 0:
            #pred = mem_label[tar_idx].long()

            beta = 0.8
            py, y_prime = F.softmax(outputs_test, dim=-1).max(1)
            flag = py > beta
            classifier_loss = F.cross_entropy(outputs_test[flag], y_prime[flag])

            classifier_loss *= args.cls_par
        else:
            classifier_loss = torch.tensor(0.0).cuda()
        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss += gentropy_loss
            im_loss = entropy_loss * args.ent_par
            # SHOT-IM
            classifier_loss = im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            if args.paradigm == 'MI':
                acc_t_te, y_pred = cal_acc(dset_loaders["Target"], netF, netC, args=args)
                #acc_t_te, y_pred = cal_auc(dset_loaders["Target-Imbalanced"], netF, netC, args=args)
                log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, iter_num, max_iter, acc_t_te)
            args.log.record(log_str)
            print(log_str)
            netF.train()


    print('Test Acc = {:.2f}%'.format(acc_t_te))

    with open('./logs/' + str(args.method) + "_pred.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(y_pred.numpy())

    gc.collect()
    torch.cuda.empty_cache()

    return acc_t_te


if __name__ == '__main__':

    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']

    dct = pd.DataFrame(
        columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                 's12', 's13'])

    for data_name in data_name_list:

        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, lr=0.001, lr_decay1=0.1, lr_decay2=1.0,
                                  ent=True, gent=True, cls_par=0, ent_par=1.0, epsilon=1e-05, layer='wn', interval=5,
                                  trial_num=trial_num, time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, smooth=0, threshold=0, distance='cosine',
                                  cov_type='oas', paradigm=paradigm, data_name=data_name)

        args.method = 'SHOT'
        #args.method = 'SHOT-IM'
        args.backbone = 'EEGNet'

        # whether to use EA
        args.align = True

        # train batch size
        args.batch_size = 32
        if paradigm == 'ERP':
            args.batch_size = 256

        # training epochs
        # 0 means use pretrained models from dnn.py for SFUDA
        args.max_epoch = 0

        # GPU device id
        try:
            device_id = str(sys.argv[1])
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
        except:
            args.data_env = 'local'

        total_acc = []

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

            sub_acc_all = np.zeros(N)
            for idt in range(N):
                args.idt = idt
                source_str = 'Except_S' + str(idt + 1)
                target_str = 'S' + str(idt + 1)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                print(info_str)
                my_log.record(info_str)
                args.log = my_log

                sub_acc_all[idt] = train_target(args)
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            total_acc.append(sub_acc_all)

            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record(acc_sub_str)
            args.log.record(acc_mean_str)

        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)

        print(str(total_acc))

        args.log.record(str(total_acc))

        subject_mean = np.round(np.average(total_acc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_acc)), 5)
        total_std = np.round(np.std(np.average(total_acc, axis=1)), 5)

        print(subject_mean)
        print(total_mean)
        print(total_std)

        args.log.record(str(subject_mean))
        args.log.record(str(total_mean))
        args.log.record(str(total_std))

        result_dct = {'dataset': data_name, 'avg': total_mean, 'std': total_std}
        for i in range(len(subject_mean)):
            result_dct['s' + str(i)] = subject_mean[i]

        dct = dct.append(result_dct, ignore_index=True)

    # save results to csv
    dct.to_csv('./logs/' + str(args.method) + ".csv")