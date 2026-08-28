import os
import torch
import numpy as np
from scipy.signal import hilbert
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


def generate_augmented_inputs(x, y, args):
    aug_list = []
    aug_list.append(identity_aug_for_tta(x, y, args))
    x_c = np.transpose(x, (0, 2, 1))
    aug_list.append(data_noise_f_for_tta(x_c, y, args))
    aug_list.append(data_mult_f_for_tta(x_c, y, args, mult_mod=0.1))
    aug_list.append(data_mult_f_for_tta(x_c, y, args, mult_mod=-0.1))
    aug_list.append(data_mult_f_for_tta(x_c, y, args, mult_mod=-0.2))
    aug_list.append(freq_mod_f_for_tta(x_c, y, args, flag='high'))
    aug_list.append(freq_mod_f_for_tta(x_c, y, args, flag='low'))

    aug_list.append(sliding_window_augmentation_for_tta(x, y, args, no=1))
    aug_list.append(sliding_window_augmentation_for_tta(x, y, args, no=2))
    aug_list.append(sliding_window_augmentation_for_tta(x, y, args, no=3))
    aug_list.append(sliding_window_augmentation_for_tta(x, y, args, no=4))
    aug_list.append(sliding_window_augmentation_for_tta(x, y, args, no=5))

    return aug_list


# data: samples * size * n_channels
# size: int(freq * window_size)
def data_noise_f_for_tta(data, labels, args):
    new_data = []
    new_labels = []
    noise_mod_val = 2
    size = args.time_sample_num
    n_channels = args.chn

    for i in range(len(labels)):
        if labels[i] >= 0:
            stddev_t = np.std(data[i])
            rand_t = np.random.rand(data[i].shape[0], data[i].shape[1])
            rand_t = rand_t - 0.5
            to_add_t = rand_t * stddev_t / noise_mod_val
            data_t = data[i] + to_add_t
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)
    new_data_ar = np.transpose(new_data_ar, (0, 2, 1))

    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    new_data_ar = new_data_ar[:, :, :eeg_length]

    new_data_ar = torch.tensor(new_data_ar, dtype=torch.float32)
    new_labels = torch.tensor(new_labels, dtype=torch.float32)
    new_data_ar = new_data_ar.unsqueeze(1)
    new_data_ar = new_data_ar.cuda()
    new_labels = new_labels.cuda()

    return new_data_ar, new_labels


def data_mult_f_for_tta(data, labels, args, mult_mod=0.1):
    new_data = []
    new_labels = []
    mult_mod = mult_mod
    size = args.time_sample_num
    n_channels = args.chn

    for i in range(len(labels)):
        if labels[i] >= 0:
            data_t = data[i] * (1 - mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)
    new_data_ar = np.transpose(new_data_ar, (0, 2, 1))

    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    new_data_ar = new_data_ar[:, :, :eeg_length]

    new_data_ar = torch.tensor(new_data_ar, dtype=torch.float32)
    new_labels = torch.tensor(new_labels, dtype=torch.float32)
    new_data_ar = new_data_ar.unsqueeze(1)
    new_data_ar = new_data_ar.cuda()
    new_labels = new_labels.cuda()

    return new_data_ar, new_labels


def data_neg_f_for_tta(data, labels, args):
    new_data = []
    new_labels = []
    size = args.time_sample_num
    n_channels = args.chn

    for i in range(len(labels)):
        if labels[i] >= 0:
            data_t = -1 * data[i]
            data_t = data_t - np.min(data_t)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)
    new_data_ar = np.transpose(new_data_ar, (0, 2, 1))

    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    new_data_ar = new_data_ar[:, :, :eeg_length]

    new_data_ar = torch.tensor(new_data_ar, dtype=torch.float32)
    new_labels = torch.tensor(new_labels, dtype=torch.float32)
    new_data_ar = new_data_ar.unsqueeze(1)
    new_data_ar = new_data_ar.cuda()
    new_labels = new_labels.cuda()

    return new_data_ar, new_labels


def freq_mod_f_for_tta(data, labels, args, flag='low'):
    new_data = []
    new_labels = []
    freq_mod = 0.2
    size = args.time_sample_num
    n_channels = args.chn

    if flag=='low':
        for i in range(len(labels)):
            if labels[i] >= 0:
                low_shift = freq_shift(data[i], -freq_mod, num_channels=n_channels)
                new_data.append(low_shift)
                new_labels.append(labels[i])

    elif flag=='high':
        for i in range(len(labels)):
            if labels[i] >= 0:
                high_shift = freq_shift(data[i], freq_mod, num_channels=n_channels)
                new_data.append(high_shift)
                new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)
    new_data_ar = np.transpose(new_data_ar, (0, 2, 1))

    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    new_data_ar = new_data_ar[:, :, :eeg_length]

    new_data_ar = torch.tensor(new_data_ar, dtype=torch.float32)
    new_labels = torch.tensor(new_labels, dtype=torch.float32)
    new_data_ar = new_data_ar.unsqueeze(1)
    new_data_ar = new_data_ar.cuda()
    new_labels = new_labels.cuda()

    return new_data_ar, new_labels


def freq_shift(x, f_shift, dt=1 / 250, num_channels=22):
    shifted_sig = np.zeros((x.shape))
    len_x = len(x)
    padding_len = 2 ** nextpow2(len_x)
    padding = np.zeros((padding_len - len_x, num_channels))
    with_padding = np.vstack((x, padding))
    hilb_T = hilbert(with_padding, axis=0)
    t = np.arange(0, padding_len)
    shift_func = np.exp(2j * np.pi * f_shift * dt * t)
    for i in range(num_channels):
        shifted_sig[:, i] = (hilb_T[:, i] * shift_func)[:len_x].real
    return shifted_sig


def nextpow2(x):
    return int(np.ceil(np.log2(np.abs(x))))


def sliding_window_augmentation_for_tta(data, labels, args, no=1):
    chs = data.shape[1]
    augmented_data = []
    augmented_labels = []
    window_size = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    if args.data == 'Driving':
        stride = args.sample_rate * 0.2
    elif args.data == 'New_driving':
        stride = args.sample_rate * 0.2
    elif args.data == 'Seed':
        stride = args.sample_rate * 0.2

    for i in range(len(data)):
        trial = data[i]  # shape: [C, T]
        label = labels[i]
        T = trial.shape[1]

        for start in range(0, int(T - window_size + 1), int(stride)):
            end = start + window_size
            window = trial[:, start:end]  # shape: [C, window_size]
            if start == int(stride) * no:
                augmented_data.append(window)
                augmented_labels.append(label)

    augmented_data = np.array(augmented_data).reshape([-1, chs, window_size])
    augmented_labels = np.array(augmented_labels)

    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    augmented_data = augmented_data[:, :, :eeg_length]

    augmented_data = torch.tensor(augmented_data, dtype=torch.float32)
    augmented_labels = torch.tensor(augmented_labels, dtype=torch.float32)
    augmented_data = augmented_data.unsqueeze(1)
    augmented_data = augmented_data.cuda()
    augmented_labels = augmented_labels.cuda()

    return augmented_data, augmented_labels


def identity_aug_for_tta(data, labels, args):
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    augmented_data = data[:, :, :eeg_length]

    augmented_data = torch.tensor(augmented_data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    augmented_data = augmented_data.unsqueeze(1)
    augmented_data = augmented_data.cuda()
    labels = labels.cuda()

    return augmented_data, labels


import random
augmentations = [
    identity_aug_for_tta,  
    data_noise_f_for_tta,
    data_mult_f_for_tta,
    freq_mod_f_for_tta,
]
def random_aug(data, labels, args):    
    aug_fn = random.choice(augmentations)
    data_c = np.transpose(data, (0, 2, 1))
    if aug_fn == identity_aug_for_tta:
        aug_data, aug_labels = aug_fn(data, labels, args)
    elif aug_fn == data_noise_f_for_tta:
        aug_data, aug_labels = data_noise_f_for_tta(data_c, labels, args)
    elif aug_fn == data_mult_f_for_tta:
        paralist = [0.1, -0.1, -0.2]
        para = random.choice(paralist)
        aug_data, aug_labels = data_mult_f_for_tta(data_c, labels, args, mult_mod=para)
    elif aug_fn == freq_mod_f_for_tta:
        paralist = ['low', 'high']
        para = random.choice(paralist)
        aug_data, aug_labels = freq_mod_f_for_tta(data_c, labels, args, flag=para)
    return aug_data, aug_labels