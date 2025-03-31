# -*- coding: utf-8 -*-
# @Time    : 2025/03/24
# @Author  : Siyang Li
# @File    : ossvm.py
import numpy as np
import torch
import mne

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

import random
import json
import sys
import argparse
import os

from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, data_alignment
from utils.metrics import compute_auroc, compute_oscr, compute_closed_set_accuracy, compute_macro_f1, \
    compute_auin, compute_auout, compute_dtacc, compute_filtered_accuracy, accuracy_score


mne.set_log_level('warning')


def train_target(args):

    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)

    indices_src = np.where(np.isin(y_src, args.src_class))

    X_src = X_src[indices_src]
    y_src = y_src[indices_src]

    indices_tar = np.where(np.isin(y_tar, args.tgt_class))

    X_tar = X_tar[indices_tar]
    y_tar = y_tar[indices_tar]

    # Identify known and unknown samples
    is_known = np.isin(y_tar, args.src_class)  # True for known classes (0, 1), False for unknown (3)

    # For unknown samples, set their true labels to -1 (unknown class)
    y_tar[~is_known] = -1

    print('after subset to open-set problem:\nX_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)

    if args.align:
        # TODO assumes equal number of trials of each source subject
        X_src = data_alignment(X_src, args.N - 1, args)
        X_tar = data_alignment(X_tar, 1, args)

    csp = mne.decoding.CSP(n_components=6)

    X_src = csp.fit_transform(X_src, y_src)
    X_tar = csp.transform(X_tar)

    # Predict with open set handling
    def predict_open_set_svm(clf, X_tar, novel_class=-1, threshold=None):

        probas = clf.decision_function(X_tar)  # Decision scores
        y_pred_original = np.argmax(probas, axis=1)  # Assign class with highest score
        max_scores = np.max(probas, axis=1)

        # Apply rejection threshold: if confidence is too low, assign "unknown" (-1)
        pred = y_pred_original.copy()
        pred[max_scores < threshold] = -1  # -1 denotes open set rejection

        return pred, probas, y_pred_original


    if len(args.src_class) > 2:
        # Train multi-class SVM (One-vs-Rest)
        np.unique(y_src)
        svm = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
        svm.fit(X_src, y_src)

        # Predict on target data
        y_pred, y_prob, y_pred_original = predict_open_set_svm(svm, X_tar, novel_class=-1, threshold=args.threshold)
    elif len(args.src_class) == 2:
        # Train SVM (One-vs-One by default in SVC)
        svm = SVC(kernel='rbf', probability=True)  # Enable probability estimates
        svm.fit(X_src, y_src)

        # Predict on target data
        y_pred_original = svm.predict(X_tar)
        y_prob = svm.predict_proba(X_tar)

        # Define Open Set thresholding: Reject samples with low confidence
        threshold = args.threshold  # Adjust based on validation or heuristics
        max_probs = np.max(y_prob, axis=1)

        y_pred = np.where(max_probs < threshold, -1, y_pred_original)  # Assign novel class if below threshold
    else:
        print('error in classes!')
        sys.exit(1)

    # Compute confidence scores (max probability for each sample)
    y_confidence = np.max(y_prob, axis=1)

    # Compute metrics
    auroc_score = compute_auroc(y_confidence, is_known)
    oscr_score = compute_oscr(y_confidence, is_known, y_tar, y_pred)
    closed_set_acc_score = compute_closed_set_accuracy(y_pred, y_tar, is_known)
    macro_f1 = compute_macro_f1(y_tar, y_pred)
    auin_score = compute_auin(y_confidence, is_known)
    auout_score = compute_auout(y_confidence, is_known)
    dtacc_score = compute_dtacc(y_confidence, is_known, y_tar, y_pred)
    filtered_accuracy = compute_filtered_accuracy(y_tar, y_pred, args.src_class, novel_class_label=-1)

    print("AUROC: {:.2f}".format(auroc_score))
    print("OSCR: {:.2f}".format(oscr_score))
    print("Closed Set Accuracy: {:.2f}".format(closed_set_acc_score))
    print("Macro-F1: {:.2f}".format(macro_f1))
    print("AUIN: {:.2f}".format(auin_score))
    print("AUOUT: {:.2f}".format(auout_score))
    print("DTACC: {:.2f}".format(dtacc_score))
    print("filtered_accuracy: {:.2f}".format(filtered_accuracy))

    return auroc_score, oscr_score, closed_set_acc_score, macro_f1, auin_score, auout_score, dtacc_score, filtered_accuracy


if __name__ == '__main__':

    data_name_list = ['BNCI2014001-4']

    for data_name in data_name_list:

        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, trial_num=trial_num,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=data_name)

        args.method = 'OSSVM'
        args.backbone = '-'

        # percentile threshold for multi-class open-set SVM
        args.threshold = 0.5

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
                auroc_score, oscr_score, closed_set_acc_score, macro_f1, auin_score, dtacc_score = train_target(
                    args)
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