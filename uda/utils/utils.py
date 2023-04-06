# -*- coding: utf-8 -*-
# @Time    : 2022/08/11
# @Author  : Siyang Li
# @File    : utils.py
import os.path as osp
import os
import math
import numpy as np
import random
import sys

import sklearn.metrics.pairwise
import torch as tr
import torch.nn as nn
import torch.utils.data
import torch.utils.data as Data
import moabb
import mne
import copy
import time
import scipy
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics.pairwise import cosine_distances

try:
    from alg_utils import EA
    from loss import ClassConfusionLoss, Entropy
    from models.cotta import CoTTA
    from models.sar import SAR, SAM
    import models.sar as sar
except:
    from utils.alg_utils import EA
    from models.cotta import CoTTA
    from utils.loss import ClassConfusionLoss, Entropy
    from models.sar import SAR, SAM
    import models.sar as sar

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2014008, BNCI2014009, BNCI2015003, BNCI2015004, EPFLP300, BNCI2014004, BNCI2015001
from moabb.paradigms import MotorImagery, P300
from sklearn.metrics import roc_auc_score



def split_data(data, axis, times):
    # Splitting data into multiple sections. data: (trials, channels, time_samples)
    data_split = np.split(data, indices_or_sections=times, axis=axis)
    return data_split


def dataset_to_file(dataset_name, data_save):
    moabb.set_log_level("ERROR")
    if dataset_name == 'BNCI2014001':
        dataset = BNCI2014001()
        paradigm = MotorImagery(n_classes=4)
        # (5184, 22, 1001) (5184,) 250Hz 9subjects * 4classes * (72+72)trials for 2sessions
    elif dataset_name == 'BNCI2014002':
        dataset = BNCI2014002()
        paradigm = MotorImagery(n_classes=2)
        # (2240, 15, 2561) (2240,) 512Hz 14subjects * 2classes * (50+30)trials * 2sessions(not namely separately)
    elif dataset_name == 'BNCI2014004':
        dataset = BNCI2014004()
        paradigm = MotorImagery(n_classes=2)
        # (6520, 3, 1126) (6520,) 250Hz 9subjects * 2classes * (?)trials * 5sessions
    elif dataset_name == 'BNCI2015001':
        dataset = BNCI2015001()
        paradigm = MotorImagery(n_classes=2)
        # (5600, 13, 2561) (5600,) 512Hz 12subjects * 2 classes * (200 + 200 + (200 for Subj 8/9/10/11)) trials * (2/3)sessions
    elif dataset_name == 'MI1':
        info = None
        return info
        # (1400, 59, 300) (1400,) 100Hz 7subjects * 2classes * 200trials * 1session
    elif dataset_name == 'BNCI2015004':
        dataset = BNCI2015004()
        paradigm = MotorImagery(n_classes=2)
        # [160, 160, 160, 150 (80+70), 160, 160, 150 (80+70), 160, 160]
        # (1420, 30, 1793) (1420,) 256Hz 9subjects * 2classes * (80+80/70)trials * 2sessions
    elif dataset_name == 'BNCI2014008':
        dataset = BNCI2014008()
        paradigm = P300()
        # (33600, 8, 257) (33600,) 256Hz 8subjects 4200 trials * 1session
    elif dataset_name == 'BNCI2014009':
        dataset = BNCI2014009()
        paradigm = P300()
        # (17280, 16, 206) (17280,) 256Hz 10subjects 1728 trials * 3sessions
    elif dataset_name == 'BNCI2015003':
        dataset = BNCI2015003()
        paradigm = P300()
        # (25200, 8, 206) (25200,) 256Hz 10subjects 2520 trials * 1session
    elif dataset_name == 'EPFLP300':
        dataset = EPFLP300()
        paradigm = P300()
        # (25200, 8, 206) (25200,) 256Hz 10subjects 1session
    elif dataset_name == 'ERN':
        ch_names = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2',
                    'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3',
                    'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'POz', 'P08', 'O1', 'O2']
        info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types=['eeg'] * 56)
        return info
        # (5440, 56, 260) (5440,) 200Hz 16subjects 1session
    # SEED (152730, 62, 5*DE*)  (152730,) 200Hz 15subjects 3sessions

    if data_save:
        print('preparing data...')
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:])
        ar_unique, cnts = np.unique(labels, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts)
        print(X.shape, labels.shape)
        np.save('./data/' + dataset_name + '/X', X)
        np.save('./data/' + dataset_name + '/labels', labels)
        meta.to_csv('./data/' + dataset_name + '/meta.csv')
    else:
        if isinstance(paradigm, MotorImagery):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:], return_epochs=True)
            return X.info
        elif isinstance(paradigm, P300):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:], return_epochs=True)
            return X.info


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def fix_random_seed(SEED):
    tr.manual_seed(SEED)
    tr.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def create_folder(dir_name, data_env, win_root):
    if not osp.exists(dir_name):
        os.system('mkdir -p ' + dir_name)
    if not osp.exists(dir_name):
        if data_env == 'gpu':
            os.mkdir(dir_name)
        elif data_env == 'local':
            os.makedirs(win_root + dir_name)


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def lr_scheduler_full(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def cal_acc(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1].float()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    accuracy = accuracy_score(true, pred)

    return accuracy * 100, pred


def cal_bca(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1].float()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    bca = balanced_accuracy_score(true, pred)
    return bca * 100, pred


def cal_auc(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1].float()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    #_, predict = tr.max(all_output, 1)
    #pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    pred = all_output[:, 1].detach().numpy()
    auc = roc_auc_score(true, pred)
    return auc * 100, pred


def cal_acc_comb(loader, model, flag=True, fc=None):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            #inputs = inputs.cuda()
            inputs = inputs
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred)

    return acc * 100, pred


def cal_acc_online(loader, model, args, balanced=True, flag=True, fc=None):
    y_true = []
    y_pred = []
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            if i == 0:
                data_cum = inputs.float().cpu()
                continue
            else:
                data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)

            if args.align:
                inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i]
                inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            outputs = outputs.float().cpu()
            labels = labels.float().cpu()
            _, predict = tr.max(outputs, 1)
            pred = tr.squeeze(predict).float()
            y_pred.append(pred.item())
            y_true.append(labels.item())

            if i == 1:
                inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[0]
                inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

                inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
                if flag:
                    _, outputs = model(inputs)
                else:
                    if fc is not None:
                        outputs, _ = model(inputs)  # modified
                    else:
                        outputs = model(inputs)
                outputs = outputs.float().cpu()
                _, predict = tr.max(outputs, 1)
                pred = tr.squeeze(predict).float()
                y_pred.append(pred.item())
                y_true.append(labels.item())

    if balanced:
        score = accuracy_score(y_true, y_pred)
    else:
        score = balanced_accuracy_score(y_true, y_pred)

    return score * 100


def cal_acc_online_testBN(loader, model, args, balanced=True, flag=True, fc=None):
    y_true = []
    y_pred = []

    optimizer = torch.optim.Adam(list(model[0].block1[2].parameters()) + list(model[0].block1[4].parameters()) + list(model[0].block2[3].parameters()), lr=args.lr)

    iter_test = iter(loader)
    for i in range(len(loader)):
        model.eval()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        if i == 0:
            data_cum = inputs.float().cpu()
            labels_cum = labels.float().cpu()
            continue
        else:
            data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            labels_cum = tr.cat((labels_cum, labels.float().cpu()), 0)

        if args.align:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

        inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
        if flag:
            _, outputs = model(inputs)
        else:
            if fc is not None:
                outputs, _ = model(inputs)  # modified
            else:
                outputs = model(inputs)
        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = tr.max(outputs, 1)
        pred = tr.squeeze(predict).float()
        y_pred.append(pred.item())
        y_true.append(labels.item())

        if i == 1:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[0]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            outputs = outputs.float().cpu()
            _, predict = tr.max(outputs, 1)
            pred = tr.squeeze(predict).float()
            y_pred.append(pred.item())
            y_true.append(labels.item())

        if (i + 1) % args.test_batch == 0:
            model[0].block1[2].train()
            model[0].block1[4].train()
            model[0].block2[3].train()

            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i-args.test_batch+1:i+1]
            inputs = inputs.reshape(args.test_batch, 1, inputs.shape[1], inputs.shape[2])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

            for step in range(args.steps):

                if flag:
                    _, outputs = model(inputs)
                else:
                    if fc is not None:
                        outputs, _ = model(inputs)  # modified
                    else:
                        outputs = model(inputs)

                optimizer.zero_grad()

                outputs = outputs.float().cpu()
                criterion = nn.CrossEntropyLoss()
                classifier_loss = criterion(outputs, labels_cum[i-args.test_batch+1:i+1].reshape(-1,).to(tr.long))
                classifier_loss.backward()  # fake backprop, only updates BN

                optimizer.step()

            model[0].block1[2].eval()
            model[0].block1[4].eval()
            model[0].block2[3].eval()
            model.eval()

    if balanced:
        score = accuracy_score(y_true, y_pred)
    else:
        score = balanced_accuracy_score(y_true, y_pred)

    return score * 100


def cal_acc_TENT(loader, model, args, balanced=True, flag=True, fc=None):
    y_true = []
    y_pred = []

    optimizer = torch.optim.Adam(list(model[0].block1[2].parameters()) + list(model[0].block1[4].parameters()) + list(model[0].block2[3].parameters()), lr=args.lr)
    #optimizer = torch.optim.SGD(list(model[0].block1[2].parameters()) + list(model[0].block1[4].parameters()) + list(model[0].block2[3].parameters()), lr=args.lr, momentum=0.9)

    model[0].block1[2].track_running_stats = False
    model[0].block1[4].running_mean = None
    model[0].block2[3].running_var = None

    iter_test = iter(loader)
    for i in range(len(loader)):
        #model.eval()
        model.train()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        if i == 0:
            data_cum = inputs.float().cpu()
            labels_cum = labels.float().cpu()
            continue
        else:
            data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            labels_cum = tr.cat((labels_cum, labels.float().cpu()), 0)

        if args.align:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

        inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
        if flag:
            _, outputs = model(inputs)
        else:
            if fc is not None:
                outputs, _ = model(inputs)  # modified
            else:
                outputs = model(inputs)
        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = tr.max(outputs, 1)
        pred = tr.squeeze(predict).float()
        y_pred.append(pred.item())
        y_true.append(labels.item())

        if i == 1:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[0]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            outputs = outputs.float().cpu()
            _, predict = tr.max(outputs, 1)
            pred = tr.squeeze(predict).float()
            y_pred.append(pred.item())
            y_true.append(labels.item())

        if (i + 1) % args.test_batch == 0:
            model[0].block1[2].train()
            model[0].block1[4].train()
            model[0].block2[3].train()

            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i-args.test_batch+1:i+1]
            inputs = inputs.reshape(args.test_batch, 1, inputs.shape[1], inputs.shape[2])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

            for step in range(args.steps):

                if flag:
                    _, outputs = model(inputs)
                else:
                    if fc is not None:
                        outputs, _ = model(inputs)  # modified
                    else:
                        outputs = model(inputs)

                optimizer.zero_grad()

                outputs = outputs.float().cpu()
                softmax_out = nn.Softmax(dim=1)(outputs)
                loss = Entropy(softmax_out)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()

            model[0].block1[2].eval()
            model[0].block1[4].eval()
            model[0].block2[3].eval()
            #model.eval()

    if balanced:
        score = accuracy_score(y_true, y_pred)
    else:
        score = balanced_accuracy_score(y_true, y_pred)

    return score * 100


def cal_acc_T3A(loader, model, args, balanced=True, flag=True, fc=None, weights=None):
    y_true = []
    y_pred = []
    results = []
    ents = []

    feature_dim = len(weights[0][0])
    # class prototypes, initialized with FC layer weights
    protos = weights

    a = np.array([-1])
    b = np.array([-1])
    # entropy records
    ent_records = [a, b]

    # size of support set
    M = 10  # T3A

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    iter_test = iter(loader)
    for i in range(len(loader)):
        model.eval()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        if i == 0:
            data_cum = inputs.float().cpu()
            labels_cum = labels.float().cpu()
            continue
        else:
            data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            labels_cum = tr.cat((labels_cum, labels.float().cpu()), 0)

        if args.align:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

        inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
        if flag:
            features_test, outputs = model(inputs)
        else:
            if fc is not None:
                outputs, _ = model(inputs)  # modified
            else:
                outputs = model(inputs)

        softmax_out = nn.Softmax(dim=1)(outputs)
        ent = Entropy(softmax_out)
        ents.append(np.round(ent.item(), 4))

        if len(protos[0]) == 1:
            prototype0 = protos[0][0]
        else:
            prototype0 = torch.mean(torch.stack(protos[0]), dim=0)
        if len(protos[1]) == 1:
            prototype1 = protos[1][0]
        else:
            prototype1 = torch.mean(torch.stack(protos[1]), dim=0)
        curr_protos = torch.stack((prototype0, prototype1))
        outputs = torch.mm(features_test, curr_protos.T.cuda())

        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = tr.max(outputs, 1)
        pred = tr.squeeze(predict).float()

        id_ = int(pred)

        if len(ent_records[id_]) < M:
            ent_records[id_] = np.append(ent_records[id_], np.round(ent.cpu().item(), 4))
            protos[id_].append(features_test.reshape(feature_dim).cpu())
        else:  # remove highest entropy term
            ind = np.argmax(ent_records[id_])
            max_ent = np.max(ent_records[id_])
            if ent < max_ent:
                ent_records[id_] = np.delete(ent_records[id_], ind)
                del protos[id_][ind]
                ent_records[id_] = np.append(ent_records[id_], np.round(ent.cpu().item(), 4))
                protos[id_].append(features_test.reshape(feature_dim).cpu())

        y_pred.append(pred.item())
        y_true.append(labels.item())

        if pred.item() == labels.item():
            results.append(1)
        else:
            results.append(0)

        # handle very first test sample, postponed due to EA
        if i == 1:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[0]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
            if flag:
                features_test, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)

            if len(protos[0]) == 1:
                prototype0 = protos[0][0]
            else:
                prototype0 = torch.mean(torch.stack(protos[0]), dim=0)
            if len(protos[1]) == 1:
                prototype1 = protos[1][0]
            else:
                prototype1 = torch.mean(torch.stack(protos[1]), dim=0)
            curr_protos = torch.stack((prototype0, prototype1))

            softmax_out = nn.Softmax(dim=1)(outputs)
            ent = Entropy(softmax_out)
            ents.append(np.round(ent.item(), 4))

            outputs = torch.mm(features_test, curr_protos.T.cuda())

            outputs = outputs.float().cpu()
            _, predict = tr.max(outputs, 1)
            pred = tr.squeeze(predict).float()

            ent_records[id_] = np.append(ent_records[id_], np.round(ent.cpu().item(), 4))
            #ent_records[id_].append(np.round(ent.cpu().item(), 4))
            protos[id_].append(features_test.reshape(feature_dim).cpu())

            print(len(protos[0]), len(protos[1]))
            print(len(ent_records[0]), len(ent_records[1]))

            y_pred.append(pred.item())
            y_true.append(labels.item())
            if pred.item() == labels.item():
                results.append(1)
            else:
                results.append(0)

    if balanced:
        score = accuracy_score(y_true, y_pred)
    else:
        score = balanced_accuracy_score(y_true, y_pred)

    return score * 100


def func_right(x):
    return 36 * (1 / math.e) * pow(x, 2) - 60 * (1 / math.e) * x + 24 * (1 / math.e)


def func_left(x):
    return (36 / 25) * (1 / math.e) * pow(x, 2) - (12 / 5) * (1 / math.e) * x


def TTA(loader, model, args, balanced=True, flag=True, fc=None):
    y_true = []
    y_pred = []
    results = []
    #ents = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if 'BN-adapt' in args.method:
        optimizer = torch.optim.Adam(
            list(model[0].block1[2].parameters()) + list(model[0].block1[4].parameters()) + list(
                model[0].block2[3].parameters()), lr=args.lr)

    # RIM
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # for DELTA
    z = [1 / 2, 1 / 2]
    #z = [2 / 3, 1 / 3]

    #c0 = 4 / 5
    #c1 = 1 / 5

    #c0 = 2 / 3
    #c1 = 1 / 3

    #c0 = 1 / 2
    #c1 = 1 / 2

    c0_ids = []
    c1_ids = []

    inconf_ids = []

    iter_test = iter(loader)
    for i in range(len(loader)):

        #print('sample ', str(i), ', input')

        model.eval()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        if i == 0:
            data_cum = inputs.float().cpu()
            labels_cum = labels.float().cpu()
            continue
        else:
            data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            labels_cum = tr.cat((labels_cum, labels.float().cpu()), 0)

        start_time = time.time()
        if args.align:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

        EA_time = time.time()
        #print('sample ', str(i), ', EA finished time in ms:', np.round((EA_time - start_time) * 1000,3))

        if args.data_env != 'local':
            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
        else:
            inputs = torch.from_numpy(inputs).to(torch.float32)
        if flag:
            _, outputs = model(inputs)
        else:
            if fc is not None:
                outputs, _ = model(inputs)  # modified
            else:
                outputs = model(inputs)

        softmax_out = nn.Softmax(dim=1)(outputs)
        #ent = Entropy(softmax_out)
        #ents.append(np.round(ent.item(), 4))

        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = tr.max(outputs, 1)
        pred = tr.squeeze(predict).float()

        if balanced:
            #y_pred.append(pred.item())
            y_pred.append(nn.Softmax(dim=1)(outputs).detach().numpy())
            y_true.append(labels.item())
        else:
            y_pred.append(nn.Softmax(dim=1)(outputs).detach().numpy())
            y_true.append(labels.item())

        if pred.item() == labels.item():
            results.append(1)
        else:
            results.append(0)

        # handle very first test sample, postponed due to EA
        if i == 1:
            model.eval()
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[0]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

            if args.data_env != 'local':
                inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
            else:
                inputs = torch.from_numpy(inputs).to(torch.float32)
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)

            outputs = outputs.float().cpu()
            _, predict = tr.max(outputs, 1)
            pred = tr.squeeze(predict).float()

            if balanced:
                #y_pred.append(pred.item())
                y_pred.append(nn.Softmax(dim=1)(outputs).detach().numpy())
                y_true.append(labels.item())
            else:
                y_pred.append(nn.Softmax(dim=1)(outputs).detach().numpy())
                y_true.append(labels.item())
            if pred.item() == labels.item():
                results.append(1)
            else:
                results.append(0)

        model.train()
        #if (i + 1) % args.test_batch == 0:  # accumulative
        if (i + 1) >= args.test_batch:  # sliding
        #if False:

            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i-args.test_batch+1:i+1]
            inputs = inputs.reshape(args.test_batch, 1, inputs.shape[1], inputs.shape[2])
            start_time = time.time()
            """
            #if (i + 1) > args.test_batch:
            if len(c0_ids) >= args.test_batch // 2 and len(c1_ids) >= args.test_batch // 2:
                aligned = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))

                ids = np.concatenate([[i], c0_ids[-4:], c1_ids[-4:]]).astype(int)
                inputs = aligned[ids, :, :]

                inputs = inputs.reshape(args.test_batch + 1, 1, inputs.shape[1], inputs.shape[2])

                #print('c0_ids:', c0_ids)
                #print('c1_ids:', c1_ids)


                '''
                current_batch = aligned[i-args.test_batch+1:i+1]
                append_ids = []
                ind = -1
                # last batch ratio is biased towards class 0
                if ratio > 0:
                    if len(c1_ids) != 0:
                        while len(append_ids) != ratio:
                            try:
                                append_ids = np.append(append_ids, c1_ids[ind])
                            except:
                                ind = -1
                                append_ids = np.append(append_ids, c1_ids[ind])
                            ind -= 1
                        append_ids = append_ids.astype(int)
                        inputs = np.concatenate([current_batch, aligned[append_ids]])
                    else:
                        inputs = current_batch
                # last batch ratio is biased towards class 1
                elif ratio < 0:
                    ratio = -ratio
                    if len(c0_ids) != 0:
                        while len(append_ids) != ratio:
                            try:
                                append_ids = np.append(append_ids, c0_ids[ind])
                            except:
                                ind = -1
                                append_ids = np.append(append_ids, c0_ids[ind])
                            ind -= 1
                        append_ids = append_ids.astype(int)
                        inputs = np.concatenate([current_batch, aligned[append_ids]])
                    else:
                        inputs = current_batch
                else:
                    inputs = current_batch

                inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
                '''
            else:
                inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i-args.test_batch+1:i+1]
                inputs = inputs.reshape(args.test_batch, 1, inputs.shape[1], inputs.shape[2])
            """
            if args.data_env != 'local':
                inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
            else:
                inputs = torch.from_numpy(inputs).to(torch.float32)

            for step in range(args.steps):

                # SAR / Filtering
                #model.eval()

                if flag:
                    _, outputs = model(inputs)
                    # for ISFDA
                    #embds, outputs = model(inputs)
                else:
                    if fc is not None:
                        outputs, _ = model(inputs)  # modified
                    else:
                        outputs = model(inputs)

                optimizer.zero_grad()

                outputs = outputs.float().cpu()

                '''
                # Pseudo-label
                criterion = nn.CrossEntropyLoss()
                pseudo_labels = torch.max(outputs, dim=1)[1]
                clf_loss = criterion(outputs, pseudo_labels)
                '''
                '''
                # BN-adapt
                model[0].block1[2].train()
                model[0].block1[4].train()
                model[0].block2[3].train()

                criterion = nn.CrossEntropyLoss()
                # fake label, only updates BN, other layers are frozen
                classifier_loss = criterion(outputs,
                                            labels_cum[i - args.test_batch + 1:i + 1].reshape(-1, ).to(tr.long))

                model[0].block1[2].eval()
                model[0].block1[4].eval()
                model[0].block2[3].eval()
                '''
                '''
                # MCC
                args.t_mcc = 2  # temperature rescaling
                mcc_loss = ClassConfusionLoss(t=args.t_mcc)(outputs)
                #loss = mcc_loss
                '''
                '''
                # Entropy
                args.t = 2  # temperature rescaling
                softmax_out = nn.Softmax(dim=1)(outputs / args.t)
                loss = Entropy(softmax_out)
                loss = torch.mean(loss)
                '''

                '''
                # Filtering Conf-thresh
                args.t = 2  # temperature rescaling
                #args.t = 2 - i / 200  # adaptive temperature rescaling
                args.epsilon = 1e-5
                softmax_out = nn.Softmax(dim=1)(outputs / args.t)

                ids_conf = np.array([])
                for l in range(len(softmax_out)):
                    if softmax_out[l, 0] > 0.6 or softmax_out[l, 0] < 0.4:
                        ids_conf = np.append(ids_conf, l)
                #print(len(ids_conf))
                ids_conf = ids_conf.astype(int)
                #print('ids_conf', ids_conf)
                inputs = inputs[ids_conf, :, :, :]
                #print('inputs.shape:', inputs.shape)
                model.train()
                if flag:
                    _, outputs = model(inputs)
                    # for ISFDA
                    #embds, outputs = model(inputs)
                else:
                    if fc is not None:
                        outputs, _ = model(inputs)  # modified
                    else:
                        outputs = model(inputs)
                optimizer.zero_grad()
                outputs = outputs.float().cpu()
                '''

                # IM
                args.t = 2  # temperature rescaling
                #args.t = 2 - i / 200  # adaptive temperature rescaling
                args.epsilon = 1e-5
                softmax_out = nn.Softmax(dim=1)(outputs / args.t)

                entropy_loss = tr.mean(Entropy(softmax_out))
                msoftmax = softmax_out.mean(dim=0)
                #gentropy_loss = tr.sum(msoftmax * tr.log(msoftmax + args.epsilon))

                '''
                # SAR filtering
                ids_conf = np.array([])
                for l in range(len(softmax_out)):
                    if softmax_out[l][0] > 0.6 or softmax_out[l][0] < 0.4:
                        ids_conf = np.append(ids_conf, l)
                    else:
                        inconf_ids.append(i - args.test_batch + 1 + l)

                print(len(ids_conf))
                inputs = inputs[ids_conf]
                model.train()
                if flag:
                    _, outputs = model(inputs)
                    # for ISFDA
                    embds, outputs = model(inputs)
                else:
                    if fc is not None:
                        outputs, _ = model(inputs)  # modified
                    else:
                        outputs = model(inputs)
                optimizer.zero_grad()
                outputs = outputs.float().cpu()
                '''
                '''
                args.b = 0.01  # decay factor

                #z = [1 / 2, 1 / 2]

                #if (i + 1) % args.test_batch == 0:
                #print(c0_ids)
                #print(c1_ids)
                decay_factor = ((1 - args.b) ** (i + 1 - args.test_batch))
                #gentropy_loss = gentropy_loss / decay_factor

                #print(decay_factor)
                #for c in range(len(z)):
                    #z[c] = z[c] * args.lambda_z + msoftmax[c].cpu() * (1 - args.lambda_z)
                if (i + 1) != args.test_batch:
                    z[0] = z[0] * decay_factor + (len(c0_ids) / (len(c0_ids) + len(c1_ids))) * (1 - decay_factor)
                    z[1] = z[1] * decay_factor + (len(c1_ids) / (len(c0_ids) + len(c1_ids))) * (1 - decay_factor)
                    #print(np.round(len(c0_ids) / (len(c0_ids) + len(c1_ids)),3), np.round(len(c1_ids) / (len(c0_ids) + len(c1_ids)),3))
                    #print(len(c0_ids) / (len(c0_ids) + len(c1_ids)), len(c1_ids) / (len(c0_ids) + len(c1_ids)))
                '''
                '''
                # DELTA
                # Dynamic online re-weighting (DOT)
                pl = torch.max(softmax_out, 1)[1]
                #w = torch.zeros((args.test_batch,))
                #w_bar = torch.zeros((args.test_batch,))
                w = torch.zeros((inputs.shape[0],))
                w_bar = torch.zeros((inputs.shape[0],))
                #for b in range(args.test_batch):
                for b in range(inputs.shape[0]):
                    w[b] = 1 / (z[pl[b]] + args.epsilon)
                #for b in range(args.test_batch):
                for b in range(inputs.shape[0]):
                    w_bar[b] = args.test_batch * w[b] / torch.sum(w)
                #msoftmax_weighted = torch.mm(softmax_out.T.cpu(), torch.tensor(w_bar).to(torch.float32).reshape(args.test_batch, 1)) / args.test_batch
                msoftmax_weighted = torch.mm(softmax_out.T.cpu(), torch.tensor(w_bar).to(torch.float32).reshape(inputs.shape[0], 1)) / inputs.shape[0]

                #args.lambda_z = 0.9  # DELTA momentum

                #if (i + 1) % args.test_batch == 0:
                #for c in range(len(z)):
                    #z[c] = z[c] * args.lambda_z + msoftmax[c].cpu() * (1 - args.lambda_z)

                #entropy_loss = tr.mean(Entropy(msoftmax_weighted))
                gentropy_loss = tr.sum(msoftmax_weighted * tr.log(msoftmax_weighted + args.epsilon))

                print('before:', msoftmax)
                print('after:', msoftmax_weighted)
                '''

                # T-TIME
                # Adaptive Marginal Distribution Regularization
                # Weighting by prior class distribution
                '''
                pl = torch.max(softmax_out, 1)[1]
                class_0_num = len(torch.where(pl == 0)[0])
                class_1_num = len(torch.where(pl == 1)[0])
                
                # threshold for "fuzzy" labels， use with class dist
                for l in range(args.test_batch):
                    if softmax_out[l][0] > 0.4 and softmax_out[l][0] < 0.5:
                        class_0_num += 1
                    elif softmax_out[l][0] >= 0.5 and softmax_out[l][0] < 0.6:
                        class_1_num += 1
                
                print(class_0_num, class_1_num)
                '''
                
                norm_msoftmax = torch.tensor([0., 0.]).to(torch.float32)

                c0 = (len(c0_ids) + 4) / (len(c0_ids) + len(c1_ids) + 4)
                c1 = (len(c1_ids) + 4) / (len(c0_ids) + len(c1_ids) + 4)

                norm_msoftmax[0] = msoftmax[0] * (1 / c0) * (1 / 2)
                norm_msoftmax[1] = msoftmax[1] * (1 / c1) * (1 / 2)

                sum_msoftmax = torch.sum(norm_msoftmax)
                normed_msoftmax = torch.tensor([0., 0.]).to(torch.float32)
                normed_msoftmax[0] = norm_msoftmax[0] / sum_msoftmax
                normed_msoftmax[1] = norm_msoftmax[1] / sum_msoftmax
                #print(msoftmax)
                #print(normed_msoftmax)

                gentropy_loss = tr.sum(normed_msoftmax * tr.log(normed_msoftmax + args.epsilon))

                #args.lambda_ada = 0.9
                # on prediction probs
                #if (i + 1) % args.test_batch == 0:
                #c0 = args.lambda_ada * c0 + msoftmax[0].detach().cpu().numpy() * (1 - args.lambda_ada)
                #c1 = args.lambda_ada * c1 + msoftmax[1].detach().cpu().numpy() * (1 - args.lambda_ada)
                # on class dist
                #c0 = args.lambda_ada * c0 + (class_0_num / (class_0_num + class_1_num)) * (1 - args.lambda_ada)
                #c1 = args.lambda_ada * c1 + (class_1_num / (class_0_num + class_1_num)) * (1 - args.lambda_ada)
                #print(c0, c1)

                '''
                # ISFDA
                # Intra-class Tightening and Inter-class Separation
                # Class Center Distances based on PL
                pl = torch.max(softmax_out, 1)[1]
                class_0_ids = torch.where(pl == 0)[0]
                class_1_ids = torch.where(pl == 1)[0]

                for l in range(len(softmax_out)):
                    if softmax_out[l][0] >= 0.5 and softmax_out[l][0] < 0.6:
                        class_1_ids = torch.cat([class_1_ids, torch.tensor([l])])
                    elif softmax_out[l][1] >= 0.5 and softmax_out[l][1] < 0.6:
                        class_0_ids = torch.cat([class_0_ids, torch.tensor([l])])

                dist_loss = None
                if len(class_0_ids) == 1:
                    class_0_center = embds[class_0_ids]
                elif len(class_0_ids) == 0:
                    dist_loss = 0
                else:
                    class_0_center = torch.mean(embds[class_0_ids], dim=0)
                if len(class_1_ids) == 1:
                    class_1_center = embds[class_1_ids]
                elif len(class_1_ids) == 0:
                    dist_loss = 0
                else:
                    class_1_center = torch.mean(embds[class_1_ids], dim=0)

                if dist_loss is None:
                    cos = nn.CosineSimilarity(dim=1)
                    inter_loss = torch.sum(torch.tensor(1) - cos(embds[class_0_ids].cpu(), class_1_center.cpu().reshape(1, -1)))
                    inter_loss += torch.sum(torch.tensor(1) - cos(embds[class_1_ids].cpu(), class_0_center.cpu().reshape(1, -1)))
                    inter_loss = inter_loss / args.test_batch
                    intra_loss = torch.sum(torch.tensor(1) - cos(embds[class_0_ids].cpu(), class_0_center.cpu().reshape(1, -1)))
                    intra_loss += torch.sum(torch.tensor(1) - cos(embds[class_1_ids].cpu(), class_1_center.cpu().reshape(1, -1)))
                    intra_loss = intra_loss / args.test_batch
                    dist_loss = intra_loss - inter_loss

                im_loss = entropy_loss + gentropy_loss + dist_loss
                #args.dist_weight = 2
                #im_loss = entropy_loss + gentropy_loss + dist_loss * args.dist_weight
                '''

                #print(entropy_loss, gentropy_loss, dist_loss)

                im_loss = entropy_loss + gentropy_loss
                loss = im_loss
                #loss = entropy_loss
                #loss = classifier_loss

                loss.backward()
                optimizer.step()

            model.eval()

            if i + 1 == args.test_batch:
                args.pred_thresh = 0.7
                pl = torch.max(softmax_out, 1)[1]
                for l in range(args.test_batch):
                    if pl[l] == 0:
                        if softmax_out[l][0] > args.pred_thresh:
                            c0_ids = np.append(c0_ids, l)
                    elif pl[l] == 1:
                        if softmax_out[l][1] > args.pred_thresh:
                            c1_ids = np.append(c1_ids, l)
                    else:
                        print('ERROR in pseudo labeling!')
                        sys.exit(0)

                '''
                # initialize ratio num(class0-class1), the imbalance ratio using confident predictions in the batch
                ratio = 0
                for l in range(args.test_batch):
                    if softmax_out[l][0] >= args.pred_thresh:
                        ratio += 1
                    if softmax_out[l][1] > args.pred_thresh:
                        ratio -= 1
                '''

                #print('start_ratio:', ratio)
            else:
                # update confident prediction ids for current test sample
                pl = torch.max(softmax_out, 1)[1]
                if pl[-1] == 0:
                    if softmax_out[-1][0] > args.pred_thresh:
                        c0_ids = np.append(c0_ids, i)
                elif pl[-1] == 1:
                    if softmax_out[-1][1] > args.pred_thresh:
                        c1_ids = np.append(c1_ids, i)
                else:
                    print('ERROR in pseudo labeling!')

                '''
                # update ratio with batch without appended samples
                ratio = 0
                for l in range(args.test_batch):
                    if softmax_out[l][0] >= args.pred_thresh:
                        ratio += 1
                    if softmax_out[l][1] > args.pred_thresh:
                        ratio -= 1
                #print('updated_ratio:', ratio)
                '''

        TTA_time = time.time()
        #print('sample ', str(i), ', TTA finished in ms:', np.round((TTA_time - start_time) * 1000, 3))

        '''
        # entropy calculation
        if i != 0:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))
            inputs = inputs.reshape((i+1), 1, inputs.shape[1], inputs.shape[2])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)

            optimizer.zero_grad()

            outputs = outputs.float().cpu()
            softmax_out = nn.Softmax(dim=1)(outputs)
            ent = Entropy(softmax_out)
            ent_mean = torch.mean(ent)
            print(i, ent_mean)
        '''
        """
        # post-pred
        if args.align:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])
        if args.data_env != 'local':
            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
        else:
            inputs = torch.from_numpy(inputs).to(torch.float32)
        if flag:
            _, outputs = model(inputs)
        else:
            if fc is not None:
                outputs, _ = model(inputs)  # modified
            else:
                outputs = model(inputs)

        softmax_out = nn.Softmax(dim=1)(outputs)
        #ent = Entropy(softmax_out)
        #ents.append(np.round(ent.item(), 4))

        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = tr.max(outputs, 1)
        pred = tr.squeeze(predict).float()

        if balanced:
            #y_pred.append(pred.item())
            y_pred.append(nn.Softmax(dim=1)(outputs).detach().numpy())
            y_true.append(labels.item())
        else:
            y_pred.append(nn.Softmax(dim=1)(outputs).detach().numpy())
            y_true.append(labels.item())

        if pred.item() == labels.item():
            results.append(1)
        else:
            results.append(0)
        """
    #print('unsure:', len(inconf_ids), np.round(len(inconf_ids) / args.trial_num, 3), inconf_ids)

    if balanced:
        _, predict = tr.max(torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num), 1)
        pred = tr.squeeze(predict).float()
        score = accuracy_score(y_true, pred)
        y_pred = np.array(y_pred).reshape(-1, args.class_num)[:, 1]
    else:
        y_pred = np.concatenate(y_pred)[:, 1]
        score = roc_auc_score(y_true, y_pred)

    return score * 100, y_pred


def TTA_stack(loader, model, args, balanced=True, flag=True, fc=None):
    y_true = []
    y_pred = []
    results = []
    #ents = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # RIM
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # for DELTA
    #z = [1 / 2, 1 / 2]
    #z = [2 / 3, 1 / 3]

    #c0 = 4 / 5
    #c1 = 1 / 5

    #c0 = 2 / 3
    #c1 = 1 / 3

    #c0 = 2 / 3
    #c1 = 1 / 3

    c0_ids = []
    c1_ids = []

    iter_test = iter(loader)
    for i in range(len(loader)):

        #print('sample ', str(i), ', input')

        model.eval()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        if i == 0:
            data_cum = inputs.float().cpu()
            labels_cum = labels.float().cpu()
            continue
        else:
            data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            labels_cum = tr.cat((labels_cum, labels.float().cpu()), 0)

        start_time = time.time()
        if args.align:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

        EA_time = time.time()
        #print('sample ', str(i), ', EA finished time in ms:', np.round((EA_time - start_time) * 1000,3))

        if args.data_env != 'local':
            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
        else:
            inputs = torch.from_numpy(inputs).to(torch.float32)
        if flag:
            _, outputs = model(inputs)
        else:
            if fc is not None:
                outputs, _ = model(inputs)  # modified
            else:
                outputs = model(inputs)

        softmax_out = nn.Softmax(dim=1)(outputs)
        #ent = Entropy(softmax_out)
        #ents.append(np.round(ent.item(), 4))

        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = tr.max(outputs, 1)
        pred = tr.squeeze(predict).float()

        if balanced:
            # y_pred.append(pred.item())
            y_pred.append(nn.Softmax(dim=1)(outputs).detach().numpy())
            y_true.append(labels.item())
        else:
            y_pred.append(nn.Softmax(dim=1)(outputs).detach().numpy())
            y_true.append(labels.item())

        if pred.item() == labels.item():
            results.append(1)
        else:
            results.append(0)

        # handle very first test sample, postponed due to EA
        if i == 1:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[0]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

            if args.data_env != 'local':
                inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
            else:
                inputs = torch.from_numpy(inputs).to(torch.float32)
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)

            outputs = outputs.float().cpu()
            _, predict = tr.max(outputs, 1)
            pred = tr.squeeze(predict).float()

            if balanced:
                # y_pred.append(pred.item())
                y_pred.append(nn.Softmax(dim=1)(outputs).detach().numpy())
                y_true.append(labels.item())
            else:
                y_pred.append(nn.Softmax(dim=1)(outputs).detach().numpy())
                y_true.append(labels.item())
            if pred.item() == labels.item():
                results.append(1)
            else:
                results.append(0)
        start_time = time.time()

        model.train()

        #if (i + 1) % args.test_batch == 0:  # accumulative
        #if (i + 1) >= args.test_batch and (len(c0_ids) < math.ceil(args.test_batch * z[1]) or len(c1_ids) < math.ceil(args.test_batch * z[0])):  # sliding
        if (i + 1) == args.test_batch:

            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i-args.test_batch+1:i+1]
            inputs = inputs.reshape(args.test_batch, 1, inputs.shape[1], inputs.shape[2])
            if args.data_env != 'local':
                inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
            else:
                inputs = torch.from_numpy(inputs).to(torch.float32)

            steps = args.steps

            for step in range(steps):
                if flag:
                    _, outputs = model(inputs)
                else:
                    if fc is not None:
                        outputs, _ = model(inputs)  # modified
                    else:
                        outputs = model(inputs)
                optimizer.zero_grad()
                outputs = outputs.float().cpu()

                # IM
                args.t = 2  # temperature rescaling
                args.epsilon = 1e-5
                softmax_out = nn.Softmax(dim=1)(outputs / args.t)
                entropy_loss = tr.mean(Entropy(softmax_out))
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = tr.sum(msoftmax * tr.log(msoftmax + args.epsilon))
                im_loss = entropy_loss + gentropy_loss

                loss = im_loss

                loss.backward()
                optimizer.step()

            args.pred_thresh = 0.7
            pl = torch.max(softmax_out, 1)[1]
            for l in range(args.test_batch):
                if pl[l] == 0:
                    if softmax_out[l][0] > args.pred_thresh:
                        c0_ids = np.append(c0_ids, l)
                elif pl[l] == 1:
                    if softmax_out[l][1] > args.pred_thresh:
                        c1_ids = np.append(c1_ids, l)
                else:
                    print('ERROR in pseudo labeling!')
                    sys.exit(0)

            # initialize ratio num(class0-class1), the imbalance ratio using confident predictions in the batch
            ratio = 0
            for l in range(args.test_batch):
                if softmax_out[l][0] >= args.pred_thresh:
                    ratio += 1
                if softmax_out[l][1] > args.pred_thresh:
                    ratio -= 1

            print('start_ratio:', ratio)

        elif (i + 1) > args.test_batch:

            aligned = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))

            current_batch = aligned[i-args.test_batch+1:i+1]
            append_ids = []
            ind = -1
            ratio = round(args.test_batch * (len(c0_ids) - len(c1_ids)) / max(len(c0_ids), len(c1_ids)))

            print('c0_ids:', c0_ids)
            print('c1_ids:', c1_ids)
            print('ratio:', ratio)

            # last batch ratio is biased towards class 0
            if ratio > 0:
                if len(c1_ids) != 0:
                    while len(append_ids) != ratio:
                        try:
                            append_ids = np.append(append_ids, c1_ids[ind])
                        except:
                            ind = -1
                            append_ids = np.append(append_ids, c1_ids[ind])
                        ind -= 1
                    append_ids = append_ids.astype(int)
                    inputs = np.concatenate([current_batch, aligned[append_ids]])
                else:
                    inputs = current_batch
            # last batch ratio is biased towards class 1
            elif ratio < 0:
                ratio = -ratio
                if len(c0_ids) != 0:
                    while len(append_ids) != ratio:
                        try:
                            append_ids = np.append(append_ids, c0_ids[ind])
                        except:
                            ind = -1
                            append_ids = np.append(append_ids, c0_ids[ind])
                        ind -= 1
                    append_ids = append_ids.astype(int)
                    inputs = np.concatenate([current_batch, aligned[append_ids]])
                else:
                    inputs = current_batch
            else:
                inputs = current_batch

            inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])

            if args.data_env != 'local':
                inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
            else:
                inputs = torch.from_numpy(inputs).to(torch.float32)

            for step in range(args.steps):

                if flag:
                    _, outputs = model(inputs)
                    # for ISFDA
                    #embds, outputs = model(inputs)
                else:
                    if fc is not None:
                        outputs, _ = model(inputs)  # modified
                    else:
                        outputs = model(inputs)

                optimizer.zero_grad()

                outputs = outputs.float().cpu()

                # IM
                args.t = 2  # temperature rescaling
                args.epsilon = 1e-5
                softmax_out = nn.Softmax(dim=1)(outputs / args.t)

                entropy_loss = tr.mean(Entropy(softmax_out))
                msoftmax = softmax_out.mean(dim=0)

                gentropy_loss = tr.sum(msoftmax * tr.log(msoftmax + args.epsilon))

                im_loss = entropy_loss + gentropy_loss
                loss = im_loss

                loss.backward()
                optimizer.step()

            '''
            # update ratio with batch without appended samples
            ratio = 0
            for l in range(args.test_batch):
                if softmax_out[l][0] >= args.pred_thresh:
                    ratio += 1
                if softmax_out[l][1] > args.pred_thresh:
                    ratio -= 1
            print('updated_ratio:', ratio)
            '''
            # update confident prediction ids for current test sample
            pl = torch.max(softmax_out, 1)[1]
            if pl[-1] == 0:
                if softmax_out[-1][0] > args.pred_thresh:
                    c0_ids = np.append(c0_ids, i)
            elif pl[-1] == 1:
                if softmax_out[-1][1] > args.pred_thresh:
                    c1_ids = np.append(c1_ids, i)
            else:
                print('ERROR in pseudo labeling!')

            model.eval()

        TTA_time = time.time()
        #print('sample ', str(i), ', TTA finished in ms:', np.round((TTA_time - start_time) * 1000, 3))

    if balanced:
        _, predict = tr.max(torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num), 1)
        pred = tr.squeeze(predict).float()
        score = accuracy_score(y_true, pred)
        y_pred = np.array(y_pred).reshape(-1, args.class_num)[:, 1]
    else:
        y_pred = np.concatenate(y_pred)[:, 1]
        score = roc_auc_score(y_true, y_pred)

    return score * 100, y_pred


def TTA_CoTTA(loader, model, args, balanced=True, flag=True, fc=None):
    y_true = []
    y_pred = []
    results = []
    ents = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    iter_test = iter(loader)
    for i in range(len(loader)):

        #print('sample ', str(i), ', input')

        model.eval()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        if i == 0:
            data_cum = inputs.float().cpu()
            labels_cum = labels.float().cpu()
            continue
        else:
            data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            labels_cum = tr.cat((labels_cum, labels.float().cpu()), 0)

        start_time = time.time()
        if args.align:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

        EA_time = time.time()
        #print('sample ', str(i), ', EA finished time in ms:', np.round((EA_time - start_time) * 1000,3))

        inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

        model.train()

        tta = CoTTA(model, optimizer)
        outputs = tta(inputs)

        #_, outputs = model(inputs)

        labels = labels.float().cpu()
        _, predict = torch.max(outputs.data, 1)

        pred = tr.squeeze(predict).float()

        y_pred.append(pred.item())
        y_true.append(labels.item())

        if pred.item() == labels.item():
            results.append(1)
        else:
            results.append(0)

        model.eval()

        # handle very first test sample, postponed due to EA
        if i == 1:
            model.train()
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[0]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
            tta = CoTTA(model, optimizer)
            outputs = tta(inputs)

            _, predict = torch.max(outputs.data, 1)
            pred = tr.squeeze(predict).float()

            y_pred.append(pred.item())
            y_true.append(labels.item())
            if pred.item() == labels.item():
                results.append(1)
            else:
                results.append(0)

            model.eval()

        start_time = time.time()

        #if (i + 1) % args.test_batch == 0:  # accumulative

        model.train()


        TTA_time = time.time()
        #print('sample ', str(i), ', TTA finished in ms:', np.round((TTA_time - start_time) * 1000, 3))

    if balanced:
        score = accuracy_score(y_true, y_pred)
    else:
        score = balanced_accuracy_score(y_true, y_pred)

    #print(results)
    #print(ents)

    return score * 100, y_pred


def TTA_SAR(loader, model, args, balanced=True, flag=True, fc=None):
    y_true = []
    y_pred = []
    results = []
    ents = []

    optimizer = torch.optim.SGD
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    opt = SAM(params, optimizer, lr=args.lr, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    iter_test = iter(loader)
    for i in range(len(loader)):

        #print('sample ', str(i), ', input')

        model.eval()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        if i == 0:
            data_cum = inputs.float().cpu()
            labels_cum = labels.float().cpu()
            continue
        else:
            data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            labels_cum = tr.cat((labels_cum, labels.float().cpu()), 0)

        start_time = time.time()
        if args.align:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

        EA_time = time.time()
        #print('sample ', str(i), ', EA finished time in ms:', np.round((EA_time - start_time) * 1000,3))

        inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

        model.train()

        tta = SAR(model, opt)
        outputs = tta(inputs)

        labels = labels.float().cpu()
        _, predict = torch.max(outputs.data, 1)

        pred = tr.squeeze(predict).float()

        y_pred.append(pred.item())
        y_true.append(labels.item())

        if pred.item() == labels.item():
            results.append(1)
        else:
            results.append(0)

        model.eval()

        # handle very first test sample, postponed due to EA
        if i == 1:
            model.train()
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[0]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
            tta = SAR(model, opt)
            outputs = tta(inputs)

            _, predict = torch.max(outputs.data, 1)
            pred = tr.squeeze(predict).float()

            y_pred.append(pred.item())
            y_true.append(labels.item())
            if pred.item() == labels.item():
                results.append(1)
            else:
                results.append(0)

            model.eval()

        start_time = time.time()

        #if (i + 1) % args.test_batch == 0:  # accumulative

        model.train()

        TTA_time = time.time()
        #print('sample ', str(i), ', TTA finished in ms:', np.round((TTA_time - start_time) * 1000, 3))

    if balanced:
        score = accuracy_score(y_true, y_pred)
    else:
        score = balanced_accuracy_score(y_true, y_pred)

    #print(results)
    #print(ents)

    return score * 100


def TTA_single(loader, model, args, balanced=True, flag=True, fc=None):
    y_true = []
    y_pred = []
    results = []
    ents = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    iter_test = iter(loader)
    for i in range(len(loader)):

        #print('sample ', str(i), ', input')

        model.eval()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        if i == 0:
            data_cum = inputs.float().cpu()
            labels_cum = labels.float().cpu()
            continue
        else:
            data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            labels_cum = tr.cat((labels_cum, labels.float().cpu()), 0)

        start_time = time.time()
        if args.align:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

        EA_time = time.time()
        #print('sample ', str(i), ', EA finished time in ms:', np.round((EA_time - start_time) * 1000,3))

        inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
        if flag:
            _, outputs = model(inputs)
        else:
            if fc is not None:
                outputs, _ = model(inputs)  # modified
            else:
                outputs = model(inputs)

        softmax_out = nn.Softmax(dim=1)(outputs)
        ent = Entropy(softmax_out)
        ents.append(np.round(ent.item(), 4))

        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = tr.max(outputs, 1)
        pred = tr.squeeze(predict).float()

        y_pred.append(pred.item())
        y_true.append(labels.item())

        if pred.item() == labels.item():
            results.append(1)
        else:
            results.append(0)

        # handle very first test sample, postponed due to EA
        if i == 1:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[0]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)

            outputs = outputs.float().cpu()
            _, predict = tr.max(outputs, 1)
            pred = tr.squeeze(predict).float()

            y_pred.append(pred.item())
            y_true.append(labels.item())
            if pred.item() == labels.item():
                results.append(1)
            else:
                results.append(0)

        start_time = time.time()

        #if (i + 1) % args.test_batch == 0:  # accumulative

        model.train()

        if (i + 1) >= args.test_batch:  # sliding


            #inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[-1]
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i-args.test_batch+1:i+1]

            #inputs = inputs.reshape(1, 1, inputs.shape[1], inputs.shape[2])
            inputs = inputs.reshape(args.test_batch, 1, inputs.shape[1], inputs.shape[2])

        #else:
        #    inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))
        #    inputs = inputs.reshape(-1, 1, inputs.shape[1], inputs.shape[2])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

            #noise = np.random.normal(0., 0.01, (inputs.shape[2], inputs.shape[3]))

            #noised_inputs = inputs + torch.from_numpy(noise).to(torch.float32).cuda()

            for step in range(args.steps):

                if flag:
                    _, outputs = model(inputs)
                else:
                    if fc is not None:
                        outputs, _ = model(inputs)  # modified
                    else:
                        outputs = model(inputs)

                optimizer.zero_grad()

                outputs = outputs.float().cpu()

                '''
                # Pseudo-label
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                pseudo_labels = torch.max(outputs, dim=1)[1]
                clf_loss = criterion(outputs, pseudo_labels)
                '''
                '''
                # MCC
                args.t_mcc = 2  # temperature rescaling
                mcc_loss = ClassConfusionLoss(t=args.t_mcc)(outputs)
                #loss = mcc_loss
                '''
                '''
                # Entropy
                args.t = 2  # temperature rescaling
                softmax_out = nn.Softmax(dim=1)(outputs / args.t)
                loss = Entropy(softmax_out)
                loss = torch.mean(loss)
                '''
                # IM
                args.t = 2  # temperature rescaling
                #args.t = 2 - i / 200  # adaptive temperature rescaling
                args.epsilon = 1e-5
                softmax_out = nn.Softmax(dim=1)(outputs / args.t)
                entropy_loss = torch.mean(Entropy(softmax_out))
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = tr.sum(msoftmax * tr.log(msoftmax + args.epsilon))

                '''
                # remove sample with biggest uncertainty in the batch from BP
                max_uncertainty_sample_id = torch.min(torch.abs(softmax_out[:, 0] - 0.5), 0)[1]
                outputs[max_uncertainty_sample_id].detach()
                softmax_out = torch.cat((softmax_out[:max_uncertainty_sample_id],softmax_out[max_uncertainty_sample_id+1:]))

                entropy_loss = Entropy(softmax_out)
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = tr.sum(msoftmax * tr.log(msoftmax + args.epsilon))

                #imbalance_level = (np.max([class_0_num, class_1_num]) / args.test_batch) * 2

                #gentropy_loss *= imbalance_level

                #im_loss = entropy_loss + gentropy_loss * trade_off
                im_loss = entropy_loss + gentropy_loss
                '''

                '''
                # self-supervision
                with torch.no_grad():
                    if flag:
                        _, noised_outputs = model(noised_inputs)
                    else:
                        if fc is not None:
                            noised_outputs, _ = model(noised_inputs)  # modified
                        else:
                            noised_outputs = model(noised_inputs)

                    noised_outputs = noised_outputs.float().cpu()
                    noised_softmax_out = nn.Softmax(dim=1)(noised_outputs / args.t)
                    noised_entropy_loss = tr.mean(Entropy(noised_softmax_out))

                diff_loss = torch.abs(noised_entropy_loss - entropy_loss)
                '''

                #reg = torch.abs(entropy_loss - torch.mean(torch.from_numpy(np.array(ents))))

                #loss = 0.2 * clf_loss + im_loss
                #loss = im_loss + reg
                loss = entropy_loss + gentropy_loss

                #print('loss', loss, entropy_loss, gentropy_loss)
                #loss = mcc_loss + im_loss

                loss.backward()
                optimizer.step()

            model.eval()


        TTA_time = time.time()
        #print('sample ', str(i), ', TTA finished in ms:', np.round((TTA_time - start_time) * 1000, 3))

        '''
        # entropy calculation
        if i != 0:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))
            inputs = inputs.reshape((i+1), 1, inputs.shape[1], inputs.shape[2])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)

            optimizer.zero_grad()

            outputs = outputs.float().cpu()
            softmax_out = nn.Softmax(dim=1)(outputs)
            ent = Entropy(softmax_out)
            ent_mean = torch.mean(ent)
            print(i, ent_mean)
        '''

    if balanced:
        score = accuracy_score(y_true, y_pred)
    else:
        score = balanced_accuracy_score(y_true, y_pred)

    #print(results)
    #print(ents)

    return score * 100, y_pred


def convert_label(labels, axis, threshold):
    # Converting labels to 0 or 1, based on a certain threshold
    label_01 = np.where(labels > threshold, 1, 0)
    #print(label_01)
    return label_01


def SML(softmax_out):
    softmax_out = torch.stack(softmax_out).cpu()
    soft = softmax_out[:, :, 1].to(torch.float32)
    soft = soft.T.detach()
    out = torch.mm(soft, soft.T)
    #hard = torch.max(softmax_out, dim=-1)[1].to(torch.float32)
    #hard = hard.T
    #out = torch.mm(hard, hard.T)
    '''
    listSVD = np.linalg.svd(out)
    u, s, v = listSVD
    accuracies = u[:, 0]
    '''
    w, v = np.linalg.eig(out)
    accuracies = v[:, 0]
    total = np.sum(accuracies)
    weights = accuracies / total
    #prediction = np.dot(weights, hard)
    prediction = np.dot(weights, soft.numpy())

    pred = convert_label(prediction, 0, 0.5)

    return pred


def TTA_ensemble(loader, models, args, balanced=True, flag=True, fc=None):
    y_true = []
    y_pred = []

    sml_pred = []

    results = []
    ents = []

    all_softmaxs = []

    optimizers = []
    for s in range(len(models)):
        optimizer = torch.optim.Adam(models[s].parameters(), lr=args.lr)
        optimizers.append(optimizer)

    iter_test = iter(loader)
    for i in range(len(loader)):

        #print('sample ', str(i), ', input')

        for s in range(len(models)):
            models[s].eval()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        if i == 0:
            data_cum = inputs.float().cpu()
            labels_cum = labels.float().cpu()
            continue
        else:
            data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            labels_cum = tr.cat((labels_cum, labels.float().cpu()), 0)

        start_time = time.time()
        if args.align:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

        EA_time = time.time()
        #print('sample ', str(i), ', EA finished time in ms:', np.round((EA_time - start_time) * 1000,3))

        inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

        softmax_out = []
        for s in range(len(models)):
            if flag:
                _, outputs = models[s](inputs)
            else:
                if fc is not None:
                    outputs, _ = models[s](inputs)  # modified
                else:
                    outputs = models[s](inputs)
            softmax = nn.Softmax(dim=1)(outputs)

            _, pred = tr.max(softmax, 1)

            softmax_out.append(softmax)
        #print(torch.stack(softmax_out).reshape(len(models), -1).shape)
        all_softmaxs.append(torch.stack(softmax_out).reshape(len(models), -1))
        #print(all_softmaxs)

        softmax_out = torch.mean(torch.stack(softmax_out).reshape(len(models), -1), 0).reshape(1, args.class_num)
        #print(softmax_out.shape)
        #ent = Entropy(softmax_out)
        #ents.append(np.round(ent.item(), 4))

        #outputs = outputs.float().cpu()
        #labels = labels.float().cpu()
        #_, predict = tr.max(outputs, 1)
        #pred = tr.squeeze(predict).float()
        _, pred = tr.max(softmax_out, 1)

        '''
        if i > len(models):
            sml_pred_curr = SML(all_softmaxs)[-1]
            sml_pred.append(sml_pred_curr)
        else:
            sml_pred.append(pred.item())
        '''

        y_pred.append(pred.item())
        y_true.append(labels.item())

        if pred.item() == labels.item():
            results.append(1)
        else:
            results.append(0)

        # handle very first test sample, postponed due to EA
        if i == 1:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[0]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

            softmax_out = []
            for s in range(len(models)):
                if flag:
                    _, outputs = models[s](inputs)
                else:
                    if fc is not None:
                        outputs, _ = models[s](inputs)  # modified
                    else:
                        outputs = models[s](inputs)
                softmax = nn.Softmax(dim=1)(outputs)

                _, pred = tr.max(softmax, 1)

                softmax_out.append(softmax)
            softmax_out = torch.mean(torch.stack(softmax_out).reshape(len(models), -1), 0).reshape(-1, args.class_num)

            #outputs = outputs.float().cpu()
            #_, predict = tr.max(outputs, 1)
            #pred = tr.squeeze(predict).float()
            pred = tr.max(softmax_out, 1)[1].float()

            sml_pred.append(pred.item())

            y_pred.append(pred.item())
            y_true.append(labels.item())
            if pred.item() == labels.item():
                results.append(1)
            else:
                results.append(0)

        start_time = time.time()

        #if (i + 1) % args.test_batch == 0:  # accumulative
        if (i + 1) >= args.test_batch:  # sliding

            for s in range(len(models)):
                models[s].train()

                inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i-args.test_batch+1:i+1]
                inputs = inputs.reshape(args.test_batch, 1, inputs.shape[1], inputs.shape[2])

                inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

                for step in range(args.steps):

                    args.t = 2  # temperature rescaling

                    if flag:
                        _, outputs = models[s](inputs)
                    else:
                        if fc is not None:
                            outputs, _ = models[s](inputs)  # modified
                        else:
                            outputs = models[s](inputs)
                        #outputs = outputs / args.t
                        #softmax = nn.Softmax(dim=1)(outputs)
                        #softmax_out.append(softmax)
                    #softmax_out = torch.mean(torch.stack(softmax_out).reshape(len(models), -1), 0).reshape(-1, args.class_num)

                    optimizers[s].zero_grad()

                    outputs = outputs.float().cpu()

                    '''
                    # Pseudo-label
                    criterion = nn.CrossEntropyLoss()
                    pseudo_labels = torch.max(outputs, dim=1)[1]
                    clf_loss = criterion(outputs, pseudo_labels)
                    '''
                    '''
                    # MCC
                    args.t_mcc = 2  # temperature rescaling
                    mcc_loss = ClassConfusionLoss(t=args.t_mcc)(outputs)
                    loss = mcc_loss
                    '''
                    '''
                    # Entropy
                    args.t = 2  # temperature rescaling
                    softmax_out = nn.Softmax(dim=1)(outputs / args.t)
                    loss = Entropy(softmax_out)
                    loss = torch.mean(loss)
                    '''
                    # IM
                    args.epsilon = 1e-5
                    softmax_out = nn.Softmax(dim=1)(outputs / args.t)
                    entropy_loss = tr.mean(Entropy(softmax_out))
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = tr.sum(msoftmax * tr.log(msoftmax + args.epsilon))
                    im_loss = entropy_loss + gentropy_loss
                    #loss = 0.2 * clf_loss + im_loss
                    loss = im_loss

                    #print('loss', loss, entropy_loss, gentropy_loss)
                    #loss = mcc_loss + im_loss

                    loss.backward()
                    #for s in range(len(models)):
                    optimizers[s].step()

            for s in range(len(models)):
                models[s].eval()

        TTA_time = time.time()
        #print('sample ', str(i), ', TTA finished in ms:', np.round((TTA_time - start_time) * 1000, 3))

        '''
        # entropy calculation
        if i != 0:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))
            inputs = inputs.reshape((i+1), 1, inputs.shape[1], inputs.shape[2])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)

            optimizer.zero_grad()

            outputs = outputs.float().cpu()
            softmax_out = nn.Softmax(dim=1)(outputs)
            ent = Entropy(softmax_out)
            ent_mean = torch.mean(ent)
            print(i, ent_mean)
        '''

    if balanced:
        score = accuracy_score(y_true, y_pred)
        #score_sml = accuracy_score(y_true, sml_pred)
    else:
        score = balanced_accuracy_score(y_true, y_pred)
        #score_sml = balanced_accuracy_score(y_true, sml_pred)

    #print(results)
    #print(ents)

    #print(score * 100)

    return score * 100
    #return score_sml * 100


def TTA_ensemble_delay(loader, models, args, balanced=True, flag=True, fc=None):
    y_true = []
    y_pred = []

    y_true_single = []
    y_pred_single = []


    results = []
    ents = []

    all_softmaxs = []

    optimizers = []
    for s in range(len(models)):
        optimizer = torch.optim.Adam(models[s].parameters(), lr=args.lr)
        optimizers.append(optimizer)

    iter_test = iter(loader)
    for i in range(len(loader)):

        #print('sample ', str(i), ', input')

        for s in range(len(models)):
            models[s].eval()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        if i == 0:
            data_cum = inputs.float().cpu()
            labels_cum = labels.float().cpu()
            continue
        else:
            data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            labels_cum = tr.cat((labels_cum, labels.float().cpu()), 0)

        start_time = time.time()
        if args.align:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

        EA_time = time.time()
        #print('sample ', str(i), ', EA finished time in ms:', np.round((EA_time - start_time) * 1000,3))

        inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

        softmax_out = []
        for s in range(len(models)):
            if flag:
                _, outputs = models[s](inputs)
            else:
                if fc is not None:
                    outputs, _ = models[s](inputs)  # modified
                else:
                    outputs = models[s](inputs)
            softmax = nn.Softmax(dim=1)(outputs)

            _, pred = tr.max(softmax, 1)
            y_pred_single.append(pred.item())
            y_true_single.append(labels.item())

            softmax_out.append(softmax)
        #print(torch.stack(softmax_out).reshape(len(models), -1).shape)
        #all_softmaxs.append(torch.max(torch.mean(torch.stack(softmax_out).reshape(len(models), -1), 0).reshape(1, args.class_num), 1)[1])
        #if i > 2:
        #    sml_pred = SML(all_softmaxs)

        softmax_out = torch.mean(torch.stack(softmax_out).reshape(len(models), -1), 0).reshape(1, args.class_num)
        #print(softmax_out.shape)
        #ent = Entropy(softmax_out)
        #ents.append(np.round(ent.item(), 4))

        #outputs = outputs.float().cpu()
        #labels = labels.float().cpu()
        #_, predict = tr.max(outputs, 1)
        #pred = tr.squeeze(predict).float()
        _, pred = tr.max(softmax_out, 1)

        y_pred.append(pred.item())
        y_true.append(labels.item())

        if pred.item() == labels.item():
            results.append(1)
        else:
            results.append(0)

        # handle very first test sample, postponed due to EA
        if i == 1:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[0]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

            softmax_out = []
            for s in range(len(models)):
                if flag:
                    _, outputs = models[s](inputs)
                else:
                    if fc is not None:
                        outputs, _ = models[s](inputs)  # modified
                    else:
                        outputs = models[s](inputs)
                softmax = nn.Softmax(dim=1)(outputs)

                _, pred = tr.max(softmax, 1)
                y_pred_single.append(pred.item())
                y_true_single.append(labels.item())

                softmax_out.append(softmax)
            softmax_out = torch.mean(torch.stack(softmax_out).reshape(len(models), -1), 0).reshape(-1, args.class_num)

            #outputs = outputs.float().cpu()
            #_, predict = tr.max(outputs, 1)
            #pred = tr.squeeze(predict).float()
            pred = tr.max(softmax_out, 1)[1].float()

            y_pred.append(pred.item())
            y_true.append(labels.item())
            if pred.item() == labels.item():
                results.append(1)
            else:
                results.append(0)

        start_time = time.time()

        #if (i + 1) % args.test_batch == 0:  # accumulative
        if (i + 1) >= args.test_batch:  # sliding

            softmax_outs =[]
            losses = []

            for s in range(len(models)):
                models[s].train()

                inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i-args.test_batch+1:i+1]
                inputs = inputs.reshape(args.test_batch, 1, inputs.shape[1], inputs.shape[2])

                inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

                for step in range(args.steps):

                    args.t = 2  # temperature rescaling

                    if flag:
                        _, outputs = models[s](inputs)
                    else:
                        if fc is not None:
                            outputs, _ = models[s](inputs)  # modified
                        else:
                            outputs = models[s](inputs)
                        #outputs = outputs / args.t
                        #softmax = nn.Softmax(dim=1)(outputs)
                        #softmax_out.append(softmax)
                    #softmax_out = torch.mean(torch.stack(softmax_out).reshape(len(models), -1), 0).reshape(-1, args.class_num)

                    optimizers[s].zero_grad()

                    outputs = outputs.float().cpu()

                    '''
                    # Pseudo-label
                    criterion = nn.CrossEntropyLoss()
                    pseudo_labels = torch.max(outputs, dim=1)[1]
                    clf_loss = criterion(outputs, pseudo_labels)
                    '''
                    '''
                    # MCC
                    args.t_mcc = 2  # temperature rescaling
                    mcc_loss = ClassConfusionLoss(t=args.t_mcc)(outputs)
                    loss = mcc_loss
                    '''
                    '''
                    # Entropy
                    args.t = 2  # temperature rescaling
                    softmax_out = nn.Softmax(dim=1)(outputs / args.t)
                    loss = Entropy(softmax_out)
                    loss = torch.mean(loss)
                    '''
                    # IM
                    args.epsilon = 1e-5
                    softmax_out = nn.Softmax(dim=1)(outputs / args.t)

                    softmax_outs.append(softmax_out)

                    entropy_loss = tr.mean(Entropy(softmax_out))
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = tr.sum(msoftmax * tr.log(msoftmax + args.epsilon))
                    im_loss = entropy_loss + gentropy_loss
                    #loss = 0.2 * clf_loss + im_loss
                    loss = im_loss

                    #print('loss', loss, entropy_loss, gentropy_loss)
                    #loss = mcc_loss + im_loss

                    losses.append(loss)
            #print('softmax_outs', torch.stack(softmax_outs))
            softmax_avg = torch.mean(torch.stack(softmax_outs), dim=0)
            #print('softmax_avg: ', softmax_avg)

            for s in range(len(models)):
                softmax_out = softmax_outs[s]
                #kl = scipy.stats.entropy(softmax_avg.detach().numpy(), softmax_out.detach().numpy())
                #print(kl)
                kl = np.mean(scipy.stats.entropy(softmax_avg.detach().numpy(), softmax_out.detach().numpy()))
                kl_loss = torch.from_numpy(np.array(kl))
                kl_loss.requires_grad = True
                #print(losses[s], torch.from_numpy(np.array(kl_loss)))
                loss_final = losses[s] + kl_loss
                #print('KL divergence:', kl_loss, ', loss', losses[s], ', final loss:', loss_final)
                #input('')

                loss_final.backward()
                optimizers[s].step()

            for s in range(len(models)):
                models[s].eval()


        TTA_time = time.time()
        #print('sample ', str(i), ', TTA finished in ms:', np.round((TTA_time - start_time) * 1000, 3))

        '''
        # entropy calculation
        if i != 0:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))
            inputs = inputs.reshape((i+1), 1, inputs.shape[1], inputs.shape[2])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)

            optimizer.zero_grad()

            outputs = outputs.float().cpu()
            softmax_out = nn.Softmax(dim=1)(outputs)
            ent = Entropy(softmax_out)
            ent_mean = torch.mean(ent)
            print(i, ent_mean)
        '''

    if balanced:
        score = accuracy_score(y_true, y_pred)
        score_single = accuracy_score(y_true_single, y_pred_single)
    else:
        score = balanced_accuracy_score(y_true, y_pred)
        score_single = balanced_accuracy_score(y_true_single, y_pred_single)

    #print(results)
    #print(ents)

    print(score_single * 100)

    return score * 100


def cal_acc_online_testPL(loader, model, args, balanced=True, flag=True, fc=None):
    y_true = []
    y_pred = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    iter_test = iter(loader)
    for i in range(len(loader)):
        model.eval()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        if i == 0:
            data_cum = inputs.float().cpu()
            labels_cum = labels.float().cpu()
            continue
        else:
            data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            labels_cum = tr.cat((labels_cum, labels.float().cpu()), 0)

        if args.align:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

        inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
        if flag:
            _, outputs = model(inputs)
        else:
            if fc is not None:
                outputs, _ = model(inputs)  # modified
            else:
                outputs = model(inputs)
        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = tr.max(outputs, 1)
        pred = tr.squeeze(predict).float()
        y_pred.append(pred.item())
        y_true.append(labels.item())

        if i == 1:
            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[0]
            inputs = inputs.reshape(1, 1, inputs.shape[0], inputs.shape[1])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            outputs = outputs.float().cpu()
            _, predict = tr.max(outputs, 1)
            pred = tr.squeeze(predict).float()
            y_pred.append(pred.item())
            y_true.append(labels.item())

        if (i + 1) % args.test_batch == 0:
            model.train()

            inputs = EA(data_cum.reshape(data_cum.shape[0], data_cum.shape[2], data_cum.shape[3]))[i-args.test_batch+1:i+1]
            inputs = inputs.reshape(args.test_batch, 1, inputs.shape[1], inputs.shape[2])

            inputs = torch.from_numpy(inputs).to(torch.float32).cuda()

            for step in range(args.steps):

                if flag:
                    _, outputs = model(inputs)
                else:
                    if fc is not None:
                        outputs, _ = model(inputs)  # modified
                    else:
                        outputs = model(inputs)

                optimizer.zero_grad()

                outputs = outputs.float().cpu()

                criterion = nn.CrossEntropyLoss()
                pseudo_labels = torch.max(outputs, dim=1)[1]
                loss = criterion(outputs, pseudo_labels)

                loss.backward()
                optimizer.step()

            model.eval()

    if balanced:
        score = accuracy_score(y_true, y_pred)
    else:
        score = balanced_accuracy_score(y_true, y_pred)

    return score * 100


def cal_bca_comb(loader, model, flag=True, fc=None):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    bca = balanced_accuracy_score(true, pred)

    return bca * 100, pred


def cal_auc_comb(loader, model, flag=True, fc=None):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    #_, predict = tr.max(all_output, 1)
    #pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    pred = all_output[:, 1].detach().numpy()
    auc = roc_auc_score(true, pred)

    return auc * 100, pred


def cal_metrics_ms(loader, netF, netCs, N, class_out, metrics):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            all_probs = None
            votes = None
            for i in range(N - 1):
                x = x.cuda()
                y = y.cuda()
                outputs = netF(x)
                outputs = netCs[i](outputs)
                predicted_probs = torch.nn.functional.softmax(outputs, dim=1)
                if all_probs is None:
                    all_probs = torch.zeros((x.shape[0], class_out)).cuda()
                else:
                    all_probs += predicted_probs.reshape(x.shape[0], class_out)

                _, predicted = torch.max(predicted_probs, 1)

                if votes is None:
                    votes = torch.zeros((x.shape[0], class_out)).cuda()

                for i in range(x.shape[0]):
                    votes[i, predicted[i]] += 1
            #_, predicted = torch.max(votes, 1)  # VOTING
            _, predicted = torch.max(all_probs, 1)  # PROBABILITY average
            y_true.append(y.cpu())
            y_pred.append(predicted.cpu())
    score = metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, )[0]
    return score * 100


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
    return logits


def cal_acc_msmm(loader, netF, all_centers, metrics):
    start_test = True
    # assume 2 classes
    subj_num = len(all_centers) // 2
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs_feature = netF(inputs)
            outputs = pairwise_distances_logits(outputs_feature,
                                               all_centers)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_outputs_feature = outputs_feature
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
                all_outputs_feature = tr.cat((all_outputs_feature, outputs_feature), 0)



    all_output = nn.Softmax(dim=1)(all_output)

    all_output = all_output.reshape(all_output.shape[0], 2, -1).sum(dim=2)

    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    score = metrics(true, pred)

    return score * 100


def cal_bca_ms_distance(loader, model, source_class_centers):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs_feature, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_outputs_feature = outputs_feature
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
                all_outputs_feature = tr.cat((all_outputs_feature, outputs_feature), 0)

    logits = pairwise_distances_logits(all_outputs_feature, source_class_centers)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    bca = balanced_accuracy_score(true, pred)

    return bca * 100


def cal_acc_comb_fusion(loader_k, loader_d, model, flag=True, fc=None):
    start_test = True
    with tr.no_grad():
        iter_test_k = iter(loader_k)
        iter_test_d = iter(loader_d)
        for i in range(len(loader_k)):
            k = next(iter_test_k)
            d = next(iter_test_d)
            inputs_k = k[0]
            labels_k = k[1]
            inputs_d = d[0]
            labels_d = d[1]
            inputs_k, inputs_d = inputs_k.cuda(), inputs_d.cuda()
            if flag:
                _, outputs = model((inputs_k, inputs_d))
            else:
                if fc is not None:
                    feas, outputs = model((inputs_k, inputs_d))
                    outputs = fc(feas)
                else:
                    outputs = model((inputs_k, inputs_d))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels_k.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels_k.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred)

    return acc * 100


def cal_bca_comb_fusion(loader_k, loader_d, model, flag=True, fc=None):
    start_test = True
    with tr.no_grad():
        iter_test_k = iter(loader_k)
        iter_test_d = iter(loader_d)
        for i in range(len(loader_k)):
            k = next(iter_test_k)
            d = next(iter_test_d)
            inputs_k = k[0]
            labels_k = k[1]
            inputs_d = d[0]
            labels_d = d[1]
            inputs_k, inputs_d = inputs_k.cuda(), inputs_d.cuda()
            if flag:
                _, outputs = model((inputs_k, inputs_d))
            else:
                if fc is not None:
                    feas, outputs = model((inputs_k, inputs_d))
                    outputs = fc(feas)
                else:
                    outputs = model((inputs_k, inputs_d))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels_k.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels_k.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    acc = balanced_accuracy_score(true, pred)

    return acc * 100


def cal_acc_multi(loader, netF_list, netC_list, args, weight_epoch=None, netG_list=None):
    num_src = len(netF_list)
    for i in range(len(netF_list)): netF_list[i].eval()

    if args.use_weight:
        if args.method == 'msdt':
            domain_weight = weight_epoch.detach()
            # tmp_weight = np.round(tr.squeeze(domain_weight, 0).t().cpu().detach().numpy().flatten(), 3)
            # print('\ntest domain weight: ', tmp_weight)
    else:
        domain_weight = tr.Tensor([1 / num_src] * num_src).reshape([1, num_src, 1]).cuda()

    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs, labels = data[0].cuda(), data[1]

            if args.use_weight:
                if args.method == 'decision':
                    weights_all = tr.ones(inputs.shape[0], len(args.src))
                    tmp_output = tr.zeros(len(args.src), inputs.shape[0], args.class_num)
                    for i in range(len(args.src)):
                        tmp_output[i] = netC_list[i](netF_list[i](inputs))
                        weights_all[:, i] = netG_list[i](tmp_output[i]).squeeze()
                    z = tr.sum(weights_all, dim=1) + 1e-16
                    weights_all = tr.transpose(tr.transpose(weights_all, 0, 1) / z, 0, 1)
                    weights_domain = tr.sum(weights_all, dim=0) / tr.sum(weights_all)
                    domain_weight = weights_domain.reshape([1, num_src, 1]).cuda()

            outputs_all = tr.cat([netC_list[i](netF_list[i](inputs)).unsqueeze(1) for i in range(num_src)], 1).cuda()
            preds = tr.softmax(outputs_all, dim=2)
            outputs_all_w = (preds * domain_weight).sum(dim=1).cuda()

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    _, predict = tr.max(all_output, 1)
    accuracy = tr.sum(tr.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    for i in range(len(netF_list)): netF_list[i].train()

    return accuracy * 100


def data_alignment(X, num_subjects, args):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    if args.data == 'BNCI2015003' and len(X) < 141:  # check is dataset BNCI2015003 and is downsampled and is not testset
        # upsampling for unequal distributions across subjects, i.e., each subject is upsampled to different num of trials
        print('before EA:', X.shape)
        out = []
        inds = [140, 140, 140, 140, 640, 840, 840, 840, 840, 840]
        inds = np.delete(inds, args.idt)
        for i in range(num_subjects):
            tmp_x = EA(X[np.sum(inds[:i]):np.sum(inds[:i + 1]), :, :])
            out.append(tmp_x)
        X = np.concatenate(out, axis=0)
        print('after EA:', X.shape)
    elif args.data == 'BNCI2015003' and len(X) > 25200:  # check is dataset BNCI2015003 and is upsampled
        # upsampling for unequal distributions across subjects, i.e., each subject is upsampled to different num of trials
        print('before EA:', X.shape)
        out = []
        inds = [4900, 4900, 4900, 4900, 4400, 4200, 4200, 4200, 4200, 4200]
        inds = np.delete(inds, args.idt)
        for i in range(num_subjects):
            tmp_x = EA(X[np.sum(inds[:i]):np.sum(inds[:i + 1]), :, :])
            out.append(tmp_x)
        X = np.concatenate(out, axis=0)
        print('after EA:', X.shape)
    else:
        print('before EA:', X.shape)
        out = []
        for i in range(num_subjects):
            tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
            out.append(tmp_x)
        X = np.concatenate(out, axis=0)
        print('after EA:', X.shape)
    return X


def data_loader(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    dset_loaders = {}
    train_bs = args.batch_size

    Xt_copy = Xt
    if args.align and not args.feature:
        Xs = data_alignment(Xs, args.N - 1, args)
        Xt_copy = Xt
        Xt = data_alignment(Xt, 1, args)

    #if Xs != None:
    # 随机打乱会导致训练结果偏高，不影响测试
    src_idx = np.arange(len(Ys))
    """
    if args.validation == 'random':  # for SEED
        num_train = int(0.9 * len(src_idx))
        tr.manual_seed(args.SEED)
        id_train, id_val = tr.utils.data.random_split(src_idx, [num_train, len(src_idx) - num_train])
    if args.validation == 'last':
        if args.paradigm == 'MI':  # for MI
            num_all = args.trial_num
            num_train = int(0.9 * num_all)
            id_train = np.array(src_idx).reshape(-1, num_all)[:, :num_train].reshape(1, -1).flatten()
            id_val = np.array(src_idx).reshape(-1, num_all)[:, num_train:].reshape(1, -1).flatten()

        elif args.paradigm == 'ERP':  # for ERP
            '''
            id_train = []
            id_val = []
            for i in range(args.N - 1):
                subj_Ys = Ys[args.trial_num * i: args.trial_num * (i+1)]
                indices_sorted = np.argsort(subj_Ys)
                ar_unique, cnts_class = np.unique(subj_Ys, return_counts=True)
                inds_train = np.concatenate((indices_sorted[:int(cnts_class[0] * 0.9)], indices_sorted[:int(cnts_class[1] * 0.9)]))
                inds_val = np.concatenate((indices_sorted[int(cnts_class[0] * 0.9):], indices_sorted[int(cnts_class[1] * 0.9):]))
                id_train.append(inds_train + args.trial_num * i)
                id_val.append(inds_val + args.trial_num * i)
            id_train = np.concatenate(id_train).reshape(1, -1).flatten()
            id_val = np.concatenate(id_val).reshape(1, -1).flatten()
            '''
            if args.data == 'BNCI2015003' and len(Xs) < 25200:
                inds = [140, 140, 140, 140, 640, 840, 840, 840, 840, 840]
                inds = np.delete(inds, args.idt)
                id_train, id_val = [], []
                for i in range(args.N - 1):
                    num_all = inds[i]
                    num_train = int(0.9 * num_all)
                    before_inds = int(np.sum(inds[:i]))
                    id_t = (np.arange(num_train, dtype=int) + before_inds)
                    id_v = (np.arange(num_all - num_train, dtype=int) + before_inds + num_train)
                    id_train.append(id_t)
                    id_val.append(id_v)
                id_train = np.concatenate(id_train).reshape(1, -1).flatten()
                id_val = np.concatenate(id_val).reshape(1, -1).flatten()
            elif args.data == 'BNCI2015003' and len(Xs) > 25200:
                inds = [4900, 4900, 4900, 4900, 4400, 4200, 4200, 4200, 4200, 4200]
                inds = np.delete(inds, args.idt)
                id_train, id_val = [], []
                for i in range(args.N - 1):
                    num_all = inds[i]
                    num_train = int(0.9 * num_all)
                    before_inds = int(np.sum(inds[:i]))
                    id_t = (np.arange(num_train, dtype=int) + before_inds)
                    id_v = (np.arange(num_all - num_train, dtype=int) + before_inds + num_train)
                    id_train.append(id_t)
                    id_val.append(id_v)
                id_train = np.concatenate(id_train).reshape(1, -1).flatten()
                id_val = np.concatenate(id_val).reshape(1, -1).flatten()
            else:
                num_all = args.trial_num
                num_train = int(0.9 * num_all)
                id_train = np.array(src_idx).reshape(-1, num_all)[:, :num_train].reshape(1, -1).flatten()
                id_val = np.array(src_idx).reshape(-1, num_all)[:, num_train:].reshape(1, -1).flatten()

            #ar_unique, class_counts = np.unique(Ys, return_counts=True)
            #num_samples = sum(class_counts)
            #labels = Ys  # corresponding labels of samples

            # assuming all subjects have equal number of trials and equal ratio of samples by classes
            #class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
            #weights = [class_weights[labels[i]] for i in range(int(num_samples))]
            #sampler_train = Data.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples * (args.N - 1)), replacement=True)
    """
    if args.validation == 'None':
        num_all = args.trial_num
        id_train = np.array(src_idx).reshape(-1, num_all).reshape(1, -1).flatten()
        id_val = np.array(src_idx).reshape(-1, num_all).reshape(1, -1).flatten()
    valid_Xs, valid_Ys = tr.from_numpy(Xs[id_val, :]).to(
        tr.float32), tr.from_numpy(Ys[id_val].reshape(-1,)).to(tr.long)
    valid_Xs = valid_Xs.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        valid_Xs = valid_Xs.permute(0, 3, 1, 2)

    train_Xs, train_Ys = tr.from_numpy(Xs[id_train, :]).to(
        tr.float32), tr.from_numpy(Ys[id_train].reshape(-1,)).to(tr.long)
    train_Xs = train_Xs.unsqueeze_(3).permute(0, 3, 1, 2)
    if 'EEGNet' in args.backbone:
        train_Xs = train_Xs.permute(0, 3, 1, 2)

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xs = Xs.permute(0, 3, 1, 2)

    Xt, Yt = tr.from_numpy(Xt).to(
        tr.float32), tr.from_numpy(Yt.reshape(-1,)).to(tr.long)
    Xt = Xt.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt = Xt.permute(0, 3, 1, 2)

    Xt_copy = tr.from_numpy(Xt_copy).to(
        tr.float32)
    Xt_copy = Xt_copy.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt_copy = Xt_copy.permute(0, 3, 1, 2)

    try:
        data_src = Data.TensorDataset(Xs.cuda(), Ys.cuda())
        source_tr = Data.TensorDataset(train_Xs.cuda(), train_Ys.cuda())
        source_te = Data.TensorDataset(valid_Xs.cuda(), valid_Ys.cuda())
        data_tar = Data.TensorDataset(Xt.cuda(), Yt.cuda())

        data_tar_online = Data.TensorDataset(Xt_copy.cuda(), Yt.cuda())

        sources_ms = []
        train_Xs_ms = split_data(train_Xs, axis=0, times=args.N - 1)
        train_Ys_ms = split_data(train_Ys, axis=0, times=args.N - 1)
        for i in range(args.N - 1):
            source = Data.TensorDataset(train_Xs_ms[i].cuda(), train_Ys_ms[i].cuda())
            sources_ms.append(source)
    except Exception:
        data_src = Data.TensorDataset(Xs, Ys)
        source_tr = Data.TensorDataset(train_Xs, train_Ys)
        source_te = Data.TensorDataset(valid_Xs, valid_Ys)
        data_tar = Data.TensorDataset(Xt, Yt)

        data_tar_online = Data.TensorDataset(Xt_copy, Yt)

        sources_ms = []
        train_Xs_ms = split_data(train_Xs, axis=0, times=args.N - 1)
        train_Ys_ms = split_data(train_Ys, axis=0, times=args.N - 1)
        for i in range(args.N - 1):
            source = Data.TensorDataset(train_Xs_ms[i], train_Ys_ms[i])
            sources_ms.append(source)

    # for DNN
    dset_loaders["source_tr"] = Data.DataLoader(source_tr, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["source_te"] = Data.DataLoader(source_te, batch_size=train_bs, shuffle=False, drop_last=False)

    # for DAN/DANN/CDAN/MCC
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True, drop_last=True)

    # for generating feature
    dset_loaders["Source"] = Data.DataLoader(data_src, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    # for online testing
    dset_loaders["Target-Online"] = Data.DataLoader(data_tar_online, batch_size=1, shuffle=False, drop_last=False)

    # for online imbalanced dataset
    # test
    class_0_ids = torch.where(Yt == 0)[0][:args.trial_num // 2]
    class_1_ids = torch.where(Yt == 1)[0][:args.trial_num // 4]  # single
    #class_1_ids = torch.where(Yt == 1)[0][:args.trial_num // 8]  # double
    all_ids = torch.cat([class_0_ids, class_1_ids])
    if args.data_env != 'local':
        data_tar_imb = Data.TensorDataset(Xt_copy[all_ids].cuda(), Yt[all_ids].cuda())
    else:
        data_tar_imb = Data.TensorDataset(Xt_copy[all_ids], Yt[all_ids])
    dset_loaders["Target-Online-Imbalanced"] = Data.DataLoader(data_tar_imb, batch_size=1, shuffle=True, drop_last=False)
    dset_loaders["target-Imbalanced"] = Data.DataLoader(data_tar_imb, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["Target-Imbalanced"] = Data.DataLoader(data_tar_imb, batch_size=train_bs * 3, shuffle=True, drop_last=False)

    if args.class_num == 2:
        # source
        indices = []
        for i in range(args.N - 1):
            indices.append(np.arange(args.trial_num // 2) + ((args.trial_num // 2) * i))
        indices = np.concatenate(indices, axis=0)
        class_0_ids = torch.where(Ys == 0)[0][indices]
        indices = []
        for i in range(args.N - 1):
            indices.append(np.arange(args.trial_num // 2) + ((args.trial_num // 2) * i))  # single
            #indices.append(np.arange(args.trial_num // 2) + ((args.trial_num // 4) * i))  # double
        indices = np.concatenate(indices, axis=0)
        class_1_ids = torch.where(Ys == 1)[0][indices]
        all_ids = torch.cat([class_0_ids, class_1_ids])
        if args.data_env != 'local':
            data_src_imb = Data.TensorDataset(Xs[all_ids].cuda(), Ys[all_ids].cuda())
        else:
            data_src_imb = Data.TensorDataset(Xs[all_ids], Ys[all_ids])
        dset_loaders["source-Imbalanced"] = Data.DataLoader(data_src_imb, batch_size=train_bs, shuffle=True, drop_last=True)

        # for multi-sources
        loader_arr = []
        for i in range(args.N - 1):
            loader = Data.DataLoader(sources_ms[i], batch_size=train_bs, shuffle=True, drop_last=True)
            loader_arr.append(loader)
        dset_loaders["sources"] = loader_arr

        loader_arr_S = []
        for i in range(args.N - 1):
            loader = Data.DataLoader(sources_ms[i], batch_size=train_bs, shuffle=True, drop_last=False)
            loader_arr_S.append(loader)
        dset_loaders["Sources"] = loader_arr_S

    return dset_loaders


