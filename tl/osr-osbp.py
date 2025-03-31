# -*- coding: utf-8 -*-
# @Time    : 2025/03/24
# @Author  : Siyang Li
# @File    : osbp.py
import numpy as np
import torch
import pandas as pd
import mne
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

import gc
import sys
import argparse
import os
import json

from utils.network import backbone_net
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader, data_alignment
from utils.metrics import auroc, oscr, closed_set_accuracy, macro_f1_with_unknown, auin, dtacc


mne.set_log_level('warning')


def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd), None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x ,lambd)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(args.feature_deep_dim, len(args.tgt_class))

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, representation, return_feat=False, reverse=False):
        if reverse:
            representation = grad_reverse(representation, self.lambd)

            # alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # reverse = ReverseLayerF.apply(representation, alpha)

            output = self.fc(representation)
        else:
            output = self.fc(representation)
        if return_feat:
            return representation, output
        return output


def train_target(args):

    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('before subset to open-set problem:\nX_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)

    indices_src = np.where(np.isin(y_src, args.src_class))
    X_src = X_src[indices_src]
    y_src = y_src[indices_src]

    indices_tar = np.where(np.isin(y_tar, args.tgt_class))
    X_tar = X_tar[indices_tar]
    y_tar = y_tar[indices_tar]

    print('after subset to open-set problem:\nX_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)

    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    criterion = nn.CrossEntropyLoss()

    G, _ = backbone_net(args, return_type='xy')
    C = Classifier(args)

    if args.data_env != 'local':
        G, C = G.cuda(), C.cuda()

    opt_g = optim.Adam(G.parameters(), lr=args.lr)
    opt_c = optim.Adam(C.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = int(args.max_epoch / 10) * max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0

    loss_s_cumulated = 0
    cnt_loss_s_cumulated = 0
    loss_t_cumulated = 0
    cnt_loss_t_cumulated = 0
    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = next(iter_source)

        try:
            inputs_target, _ = next(iter_target)
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, _ = next(iter_target)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1

        opt_g.zero_grad()
        opt_c.zero_grad()

        features_source = G(inputs_source)
        outputs_source = C(features_source)

        loss_s = criterion(outputs_source, labels_source)
        loss_s.backward()
        loss_s_cumulated += loss_s.item()
        cnt_loss_s_cumulated += 1

        t_tensor = torch.tensor([1 - args.t, args.t])
        temp_tensor = t_tensor.repeat(len(inputs_source), 1).cuda()
        target_funk = Variable(temp_tensor)
        p = 1.0
        C.set_lambda(p)
        features_target = G(inputs_target)
        outputs_target = C(features_target, reverse=True)
        out_t = F.softmax(outputs_target, dim=-1)

        # sum of all old classes prob
        prob1 = torch.sum(out_t[:, :len(args.tgt_class) - 1], 1).view(-1, 1)
        # new class prob
        prob2 = out_t[:, len(args.tgt_class) - 1].contiguous().view(-1, 1)

        prob = torch.cat((prob1, prob2), 1)
        loss_t = bce_loss(prob, target_funk)
        loss_t_cumulated += loss_t.item()
        cnt_loss_t_cumulated += 1

        loss_t.backward()
        opt_g.step()
        opt_c.step()
        opt_g.zero_grad()
        opt_c.zero_grad()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            G.eval()
            C.eval()
            base_network = nn.Sequential(G, C)


            loss_s_avg = loss_s_cumulated / cnt_loss_s_cumulated
            loss_s_cumulated = 0
            cnt_loss_s_cumulated = 0

            loss_t_avg = loss_t_cumulated / cnt_loss_t_cumulated
            loss_t_cumulated = 0
            cnt_loss_t_cumulated = 0

            acc, all_output = cal_acc_comb(dset_loaders["Target"], base_network, args=args, flag=False, fc=None)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}; Loss_src_CE = {:.2f}; Loss_BCE = {:.2f}'.format(args.task_str, int(iter_num // len(dset_loaders[
                                                                                                      "source"])),
                                                                                              int(max_iter // len(
                                                                                                  dset_loaders[
                                                                                                      "source"])),
                                                                                              acc, loss_s_avg, loss_t_avg)
            print(log_str)

            base_network.train()
            G.train()
            C.train()

    print(f"OSBP Accuracy (including novel class): {acc:.2f}")

    print('saving model...')

    base_network = nn.Sequential(G, C)
    if args.align:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_t' + str(args.t) + '.ckpt')
    else:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_t' + str(args.t) + '_noEA' + '.ckpt')

    _, predict = torch.max(all_output, 1)
    y_pred = torch.squeeze(predict).float().detach().cpu().numpy()
    y_score = all_output.detach().cpu().numpy()

    # Compute metrics
    auroc_score = auroc(np.isin(y_tar, args.src_class).astype(int), y_score)
    oscr_score = oscr(y_tar, y_pred, y_score, args.src_class)
    closed_set_acc_score = closed_set_accuracy(y_tar, y_pred, args.src_class)
    macro_f1 = macro_f1_with_unknown(y_tar, y_pred, args.src_class)
    auin_score = auin(y_tar, y_score, args.src_class)
    dtacc_score = dtacc(y_tar, y_score, args.src_class)
    print("AUROC: {:.2f}".format(auroc_score))
    print("OSCR: {:.2f}".format(oscr_score))
    print("Closed Set Accuracy: {:.2f}".format(closed_set_acc_score))
    print("Macro-F1: {:.2f}".format(macro_f1))
    print("AUIN: {:.2f}".format(auin_score))
    print("AUOUT: {:.2f}".format(auout(y_tar, y_score, args.src_class)))
    print("DTACC: {:.2f}".format(dtacc_score))

    return auroc_score, oscr_score, closed_set_acc_score, macro_f1, auin_score, auout_score, dtacc_score


if __name__ == '__main__':

    data_name_list = ['BNCI2014001-4']

    dct = pd.DataFrame(
        columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                 's12', 's13'])

    for data_name in data_name_list:

        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, trial_num=trial_num,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=data_name)

        args.method = 'OSBP'
        args.backbone = 'EEGNet'

        # t for OSBP
        args.t = 0.5

        args.root_dir = '/mnt/data2/sylyoung/EEG/DeepTransferEEG/'

        args.src_class = [0, 1]
        args.tgt_class = [0, 1, 3]

        # whether to use EA
        args.align = False

        # learning rate
        args.lr = 0.001

        # train batch size
        args.batch_size = 32
        if paradigm == 'ERP':
            args.batch_size = 256

        # training epochs
        args.max_epoch = 100

        # GPU device id
        try:
            device_id = str(sys.argv[1])
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
        except:
            args.data_env = 'local'

        total_auroc = []
        total_oscr = []
        total_closed_set_acc = []
        total_macro_f1 = []
        total_auin = []
        total_dtacc = []

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

            sub_auroc_score_all = np.zeros(N)
            sub_oscr_score_all = np.zeros(N)
            sub_closed_set_acc_score_all = np.zeros(N)
            sub_macro_f1_all = np.zeros(N)
            sub_auin_score_all = np.zeros(N)
            sub_dtacc_score_all = np.zeros(N)
            for idt in range(N):
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                print(info_str)
                auroc_score, oscr_score, closed_set_acc_score, macro_f1, auin_score, dtacc_score = train_target(args)
                sub_auroc_score_all[idt] = auroc_score
                sub_oscr_score_all[idt] = oscr_score
                sub_closed_set_acc_score_all[idt] = closed_set_acc_score
                sub_macro_f1_all[idt] = macro_f1
                sub_auin_score_all[idt] = auin_score
                sub_dtacc_score_all[idt] = dtacc_score
            total_auroc.append(sub_auroc_score_all)
            total_oscr.append(sub_oscr_score_all)
            total_closed_set_acc.append(sub_closed_set_acc_score_all)
            total_macro_f1.append(sub_macro_f1_all)
            total_auin.append(sub_auin_score_all)
            total_dtacc.append(sub_dtacc_score_all)

        print('\n' + '#' * 20 + 'final results' + '#' * 20)

        print('\n' + '#' * 20 + 'AUROC' + '#' * 20)
        print(str(total_auroc))
        subject_mean = np.round(np.average(total_auroc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_auroc)), 5)
        total_std = np.round(np.std(np.average(total_auroc, axis=1)), 5)
        print(subject_mean)
        print(total_mean)
        print(total_std)

        print('\n' + '#' * 20 + 'OSCR' + '#' * 20)
        print(str(total_oscr))
        subject_mean = np.round(np.average(total_oscr, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_oscr)), 5)
        total_std = np.round(np.std(np.average(total_oscr, axis=1)), 5)
        print(subject_mean)
        print(total_mean)
        print(total_std)

        print('\n' + '#' * 20 + 'Closed Set Accuracy' + '#' * 20)
        print(str(total_closed_set_acc))
        subject_mean = np.round(np.average(total_closed_set_acc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_closed_set_acc)), 5)
        total_std = np.round(np.std(np.average(total_closed_set_acc, axis=1)), 5)
        print(subject_mean)
        print(total_mean)
        print(total_std)

        print('\n' + '#' * 20 + 'Macro F1' + '#' * 20)
        print(str(total_macro_f1))
        subject_mean = np.round(np.average(total_macro_f1, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_macro_f1)), 5)
        total_std = np.round(np.std(np.average(total_macro_f1, axis=1)), 5)
        print(subject_mean)
        print(total_mean)
        print(total_std)

        print('\n' + '#' * 20 + 'AUIN' + '#' * 20)
        print(str(total_auin))
        subject_mean = np.round(np.average(total_auin, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_auin)), 5)
        total_std = np.round(np.std(np.average(total_auin, axis=1)), 5)
        print(subject_mean)
        print(total_mean)
        print(total_std)

        print('\n' + '#' * 20 + 'DTACC' + '#' * 20)
        print(str(total_dtacc))
        subject_mean = np.round(np.average(total_dtacc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_dtacc)), 5)
        total_std = np.round(np.std(np.average(total_dtacc, axis=1)), 5)
        print(subject_mean)
        print(total_mean)
        print(total_std)

    # Convert the argparse.Namespace object to a dictionary
    args_dict = vars(args)
    # Save the dictionary as a JSON file
    with open('./logs/' + str(args.method) + '-args.json', 'w') as f:
        json.dump(args_dict, f, indent=4)