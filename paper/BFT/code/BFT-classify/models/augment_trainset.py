import pywt
import numpy as np
# pyhht and PyEMD were imported here but never used, and neither package is a
# dependency, so importing this module failed before it ran a single line.
from scipy.signal import hilbert
from itertools import combinations

import random


def CR_transform(X, left_mat, right_mat):
    """
    Parameters
    ----------
    X: numpy array of shape (num_samples, num_channels, num_timesamples)
    left_mat: list or array of channel indices for the left brain hemisphere
    right_mat: list or array of channel indices for the right brain hemisphere

    Returns
    -------
    transformedX: numpy array of same shape as X with left/right channels swapped
    """
    X = X.copy()  # Avoid modifying original input
    num_samples, num_channels, num_timesamples = X.shape
    transformedX = np.zeros_like(X)

    left_mat = list(left_mat)
    right_mat = list(right_mat)

    for ch in range(num_channels):
        if ch in left_mat:
            ind = left_mat.index(ch)
            transformedX[:, ch, :] = X[:, right_mat[ind], :]
        elif ch in right_mat:
            ind = right_mat.index(ch)
            transformedX[:, ch, :] = X[:, left_mat[ind], :]
        else:
            transformedX[:, ch, :] = X[:, ch, :]

    return transformedX


def DWTAug_for_random(data, label, args, wavename='db4'):
    num_subjects = 2
    total_samples = data.shape[0]
    samples_per_subject = total_samples // num_subjects

    subject_data = np.split(data, num_subjects, axis=0)
    subject_labels = np.split(label, num_subjects, axis=0)

    aug_data_list = []
    aug_label_list = []

    for i, j in combinations(range(num_subjects), 2):
        data_i, label_i = subject_data[i], subject_labels[i]
        data_j, label_j = subject_data[j], subject_labels[j]

        for cls in [0, 1]:
            Xs = data_i[label_i == cls]
            Xt = data_j[label_j == cls]
            ys = label_i[label_i == cls]

            min_len = min(len(Xs), len(Xt))
            if min_len == 0:
                continue

            Xs = Xs[:min_len]
            Xt = Xt[:min_len]
            ys = ys[:min_len]

            ScA, ScD = pywt.dwt(Xs, wavename)
            TcA, TcD = pywt.dwt(Xt, wavename)

            Xs_aug = pywt.idwt(ScA, TcD, wavename, 'smooth')
            Xt_aug = pywt.idwt(TcA, ScD, wavename, 'smooth')

            Xs_aug = Xs_aug[:, :, :Xs.shape[-1]]
            Xt_aug = Xt_aug[:, :, :Xs.shape[-1]]

            aug_data_list.extend([Xs_aug, Xt_aug])
            aug_label_list.extend([ys, ys])  

    X_aug = np.concatenate(aug_data_list, axis=0)
    y_aug = np.concatenate(aug_label_list, axis=0)

    return X_aug, y_aug


def identity_aug(data, labels):
    return data, labels


augmentations = [
    identity_aug,  
    CR_transform,
    DWTAug_for_random,
]

def data_aug_random(data, last_data, labels, last_label, args):
    aug_fn = random.choice(augmentations)
    if aug_fn == identity_aug:
        aug_data, aug_labels = aug_fn(data, labels)
    elif aug_fn == CR_transform:
        if args.data == 'BNCI2014001':
            left_mat = [1, 2, 6, 7, 8, 13, 14, 18]
            right_mat = [5, 4, 12, 11, 10, 17, 16, 20]  
        elif args.data == 'Zhou2016':
            left_mat = [0, 2, 5, 8, 11]
            right_mat = [1, 4, 7, 10, 13]
        elif args.data == 'Schirrmeister2017':
            left_mat = [0, 3, 4, 8, 9, 12, 13, 14, 19, 20, 23, 24, 29, 32, 33, 36, 37, 40, 43, 44, 
                        47, 50, 51, 54, 55, 58, 60, 62, 64, 66, 68, 70, 72, 75, 76, 79, 80, 83, 84, 87, 
                        88, 91, 93, 96, 98, 100, 101, 104, 105, 106, 110, 111, 114, 115, 118, 119, 122, 123, 126]
            right_mat = [1, 7, 6, 11, 10, 18, 17, 16, 22, 21, 27, 26, 31, 35, 34, 39, 38, 42, 46, 45, 
                         49, 53, 52, 57, 56, 59, 61, 63, 65, 67, 69, 71, 74, 78, 77, 82, 81, 86, 85, 90, 
                         89, 92, 95, 97, 99, 103, 102, 109, 108, 107, 113, 112, 117, 116, 121, 120, 125, 124, 127]
        aug_data = aug_fn(data, left_mat, right_mat)
        aug_labels = 1 - labels
    elif aug_fn == DWTAug_for_random:
        all_data = np.concatenate((data, last_data), axis=0)
        all_labels = np.concatenate((labels, last_label), axis=0)
        aug_data, aug_labels = aug_fn(all_data, all_labels, args, wavename='db4')
        
    return aug_data, aug_labels