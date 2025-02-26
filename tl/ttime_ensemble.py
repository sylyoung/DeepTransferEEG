# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : ttime_ensemble.py
# run this file after running tl/ttime.py, which first generate model prediction probability scores
import numpy as np
import random
import pandas as pd
import torch as tr
import torch.utils.data
from sklearn.metrics import accuracy_score
from utils.dataloader import data_process
import warnings
warnings.filterwarnings("ignore")


def convert_label(labels, axis, threshold, minus1=False):
    if minus1:
        # Converting labels to -1 or 1, based on a certain threshold
        if np.random.randint(2, size=1)[0] == 1:
            label_01 = np.where(labels > threshold, 1, -1)
        else:
            label_01 = np.where(labels >= threshold, 1, -1)
    else:
        # Converting labels to 0 or 1, based on a certain threshold
        if np.random.randint(2, size=1)[0] == 1:
            label_01 = np.where(labels > threshold, 1, 0)
        else:
            label_01 = np.where(labels >= threshold, 1, 0)
    return label_01


def reverse_label(labels):
    # Reversing labels from 0 to 1, or 1 to 0
    return 1 - labels


def SML(preds):
    # corrected implementation
    """
    Parameters
    ----------
    preds : numpy array
        data of shape (num_models, num_test_samples)

    Returns
    ----------
    pred : numpy array
        data of shape (num_test_samples)
    """
    preds = convert_label(preds, 1, 0.5, minus1=True)
    mu = np.mean(preds, axis=1)
    deviations = preds - mu[:, np.newaxis]
    # Calculate the covariance matrix
    Q = np.dot(deviations, deviations.T) / (preds.shape[1] - 1)
    # Principal eigenvector
    v = np.linalg.eig(Q)[1][:, 0]
    if v[0] < 0:
        v = -v
    predictions = np.einsum('a,ab->b', v, preds)
    pred = np.where(predictions >= 0, 1, 0)
    return pred


def SML_multiclass(preds, n_classes):
    # corrected implementation using one-versus-rest method, differs from paper description
    """
    Parameters
    ----------
    preds : numpy array
        data of shape (num_models, num_test_samples, num_classes)

    n_classes: int

    Returns
    ----------
    pred : numpy array
        data of shape (num_test_samples)
    """
    preds = np.argmax(preds, -1)

    preds_one_hot = []
    for i in range(len(preds)):
        max_indices = preds[i]
        encoded_arr = np.zeros((preds.shape[1], n_classes), dtype=int)
        encoded_arr[np.arange(preds.shape[1]), max_indices] = 1
        preds_one_hot.append(encoded_arr)
    preds_one_hot = np.stack(preds_one_hot)

    preds = preds_one_hot

    weights_all = []
    class_num = preds.shape[-1]
    for i in range(class_num):

        # {-1, 1}
        pred = np.ones((preds.shape[0], preds.shape[1])) * -1
        argmax_inds = np.argmax(preds, axis=-1)
        for j in range(len(preds)):
            for n in range(len(preds[1])):
                if argmax_inds[j, n] == i:
                    pred[j, n] = 1
                else:
                    pred[j, n] = -1
        mu = np.mean(pred, axis=1)
        deviations = pred - mu[:, np.newaxis]
        # Calculate the covariance matrix
        Q = np.dot(deviations, deviations.T) / (pred.shape[1] - 1)
        # Principal eigenvector
        v = np.linalg.eig(Q)[1][:, 0]
        if v[0] <= 0:
            v = -v
        weights = v / np.sum(v)  # ensemble weights
        weights_all.append(weights)

    weights_final = np.sum(np.array(weights_all), axis=0)

    predictions = np.einsum('a,abc->bc', weights_final, preds_one_hot)
    pred = np.argmax(predictions, axis=1)

    return pred


def voting_ensemble_binary(preds):
    # preds of numpy array shape (n_classifier, n_samples), predictions are 0/1
    n_classifier, n_samples = preds.shape
    sum_votes = np.sum(preds, axis=0)
    vote_pred = convert_label(sum_votes, 0, n_classifier / 2)
    return vote_pred


def voting_ensemble_multiclass(preds, n_classes):
    # preds of numpy array shape (n_classifier, n_samples)
    n_classifier, n_samples = preds.shape
    votes_mat = np.zeros((n_classes, n_samples))
    for i in range(n_classifier):
        for j in range(n_samples):
            class_id = preds[i, j]
            votes_mat[class_id, j] += 1
    votes_pred = []
    for i in range(n_samples):
        pred = np.random.choice(np.flatnonzero(votes_mat[:, i] == votes_mat[:, i].max()))
        votes_pred.append(pred)
    votes_pred = np.array(votes_pred)
    return votes_pred


def fix_random_seed(SEED):
    tr.manual_seed(SEED)
    tr.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def binary_classification():
    method = 'T-TIME'
    data_name_list = ['BNCI2014001']
    # data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']

    for data_name in data_name_list:

        print(data_name)

        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(data_name)

        total_mean = [[], [], [], []]

        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640

        print('class_num', class_num)

        seed_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        preds = []
        for SEED in seed_arr:
            path = './logs/' + str(data_name) + '_'+ str(method) + '_seed_' + str(SEED) + "_pred.csv"
            df = pd.read_csv(path, header=None)
            preds.append(df.to_numpy())

        preds = np.stack(preds)  # (num_models, num_subjects, num_test_samples)
        print('test set preds shape:', preds.shape)

        for ens_num in range(3, 11):

            print('Ensembling of ' + str(ens_num) + ' models...')

            acc_avg = []
            acc_vote = []
            acc_sml = []

            for subj in range(num_subjects):

                pred = preds[:, subj, :]  # (num_models, num_test_samples)
                true = y[np.arange(trial_num).astype(int) + trial_num * subj]
                test_trial_num = trial_num

                # average
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_pred = np.average(pred[ens_ids, :], axis=0)
                    ens_prediction = convert_label(ens_pred, 0, 0.5)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_avg.append(seed_acc)

                # voting
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    votes = []
                    for id_ in ens_ids:
                        vote_single = convert_label(pred[id_, :], 0, 0.5)
                        votes.append(vote_single)
                    votes = np.stack(votes)
                    ens_prediction = voting_ensemble_multiclass(votes, n_classes=class_num)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_vote.append(seed_acc)

                # SML
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_prediction = []
                    for sample in range(test_trial_num):
                        if sample < ens_num:
                            ens_pred = np.average(pred[ens_ids, sample], axis=0)
                            curr_pred = convert_label(ens_pred, 0, 0.5).item()
                        else:
                            curr_table = pred[ens_ids, :sample + 1]
                            curr_pred = SML(curr_table)[-1]
                        ens_prediction.append(curr_pred)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_sml.append(seed_acc)

            method_cnt = 0
            for score in [acc_avg, acc_vote, acc_sml]:

                if method_cnt == 0:
                    print('###############Average Ensemble###############')
                if method_cnt == 1:
                    print('###############Voting Ensemble################')
                if method_cnt == 2:
                    print('###############SML Ensemble###################')
                score = np.array(score).transpose((1, 0))
                subject_mean = np.round(np.average(score, axis=0) * 100, 2)
                dataset_mean = np.round(np.average(np.average(score)) * 100, 2)
                dataset_std = np.round(np.std(np.average(score, axis=1)) * 100, 2)

                print(subject_mean)
                print(dataset_mean)
                print(dataset_std)

                total_mean[method_cnt].append(dataset_mean)

                method_cnt += 1

        print(total_mean)


def multiclass_classification():
    method = 'T-TIME'
    data_name_list = ['BNCI2014001-4']

    for data_name in data_name_list:

        print(data_name)

        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(data_name)

        total_mean = [[], [], []]

        paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248

        print('class_num', class_num)

        seed_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        preds = []
        for SEED in seed_arr:
            path = './logs/' + str(data_name) + '_' + str(method) + '_seed_' + str(SEED) + "_pred.csv"
            df = pd.read_csv(path, header=None)
            preds.append(df.to_numpy())

        preds = np.stack(preds)  # (num_models, num_subjects, num_test_samples)
        preds = preds.reshape(len(seed_arr), N, trial_num, class_num)
        print('test set preds shape:', preds.shape)

        for ens_num in range(3, 11):

            print('Ensembling of ' + str(ens_num) + ' models...')

            acc_avg = []
            acc_vote = []
            acc_smlpred = []

            for subj in range(num_subjects):

                pred = preds[:, subj, :, :]
                true = y[np.arange(trial_num).astype(int) + trial_num * subj]
                test_trial_num = trial_num

                # average
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_pred = np.average(pred[ens_ids], axis=0)
                    ens_pred = np.argmax(ens_pred, axis=-1)
                    ens_score = accuracy_score(true, ens_pred)
                    seed_acc.append(ens_score)
                acc_avg.append(seed_acc)

                # voting
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    votes = []
                    for id_ in ens_ids:
                        vote_single = np.argmax(pred[id_, :, :], axis=-1)
                        votes.append(vote_single)
                    votes = np.stack(votes)
                    ens_prediction = voting_ensemble_multiclass(votes, n_classes=class_num)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_vote.append(seed_acc)

                # SML multi-class (TTA models)
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_prediction = []
                    for sample in range(test_trial_num):
                        if sample < ens_num:
                            ens_pred = np.average(pred[ens_ids, sample, :], axis=0)
                            curr_pred = np.argmax(ens_pred, axis=-1)
                        else:
                            curr_table = pred[ens_ids, :sample + 1, :]
                            curr_pred = SML_multiclass(curr_table, class_num)[-1]
                        ens_prediction.append(curr_pred)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_smlpred.append(seed_acc)

            method_cnt = 0
            for score in [acc_avg, acc_vote, acc_smlpred]:

                if method_cnt == 0:
                    print('###############Average Ensemble###############')
                if method_cnt == 1:
                    print('###############Voting Ensemble################')
                if method_cnt == 2:
                    print('###############SML (multi-class) Ensemble###############')

                score = np.array(score).transpose((1, 0))
                subject_mean = np.round(np.average(score, axis=0) * 100, 2)
                dataset_mean = np.round(np.average(np.average(score)) * 100, 2)
                dataset_std = np.round(np.std(np.average(score, axis=1)) * 100, 2)

                print(subject_mean)
                print(dataset_mean)
                print(dataset_std)

                total_mean[method_cnt].append(dataset_mean)

                method_cnt += 1

        print(total_mean)


if __name__ == '__main__':
    fix_random_seed(42)
    binary_classification()
    multiclass_classification()



