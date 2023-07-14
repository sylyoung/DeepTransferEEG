import sys
import random

import numpy as np
import moabb
import mne

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2014008, BNCI2014009, BNCI2015003, BNCI2015004, EPFLP300, BNCI2014004, BNCI2015001
from moabb.paradigms import MotorImagery, P300


def split_data(data, axis, times):
    # Splitting data into multiple sections. data: (trials, channels, time_samples)
    data_split = np.split(data, indices_or_sections=times, axis=axis)
    return data_split


def convert_label(labels, axis, threshold):
    # Converting labels to 0 or 1, based on a certain threshold
    label_01 = np.where(labels > threshold, 1, 0)
    print(label_01)
    return label_01


def time_cut(data, cut_percentage):
    # Time Cutting: cut at a certain percentage of the time. data: (..., ..., time_samples)
    data = data[:, :, :int(data.shape[2] * cut_percentage)]
    return data


def traintest_split_cross_subject(dataset, X, y, num_subjects, test_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    test_x = data_subjects.pop(test_subject_id)
    test_y = labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    train_y = np.concatenate(labels_subjects, axis=0)
    print('Test subject s' + str(test_subject_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


def traintest_split_domain_classifier(dataset, X, y, num_subjects, test_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    data_subjects.pop(test_subject_id)
    labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    for i in range(num_subjects - 1):
        labels_subjects[i] = np.ones((int(len(labels_subjects[i]))),) * i
    train_y = np.concatenate(labels_subjects, axis=0)
    print('Test subject s' + str(test_subject_id))
    print('Training:', train_x.shape, train_y.shape)
    return train_x, train_y, None, None


def traintest_split_domain_classifier_pretest(dataset, X, y, num_subjects, ratio):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    train_x_all = []
    train_y_all = []
    test_x_all = []
    test_y_all = []
    for i in range(num_subjects):
        data = data_subjects[i]
        random.shuffle(data)
        train_x_all.append(data[:int(len(data) * ratio)])
        train_y_all.append(np.ones((int(len(data) * ratio)),) * i)
        test_x_all.append(data[int(len(data) * ratio):])
        test_y_all.append(np.ones((int(len(data) * (1 - ratio))),) * i)
    train_x = np.concatenate(train_x_all, axis=0)
    train_y = np.concatenate(train_y_all, axis=0)
    test_x = np.concatenate(test_x_all, axis=0)
    test_y = np.concatenate(test_y_all, axis=0)
    print('Training/Test split:', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y


def traintest_split_multisource(dataset, X, y, num_subjects, test_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    test_x = data_subjects.pop(test_subject_id)
    test_y = labels_subjects.pop(test_subject_id)
    train_x = data_subjects
    train_y = labels_subjects
    print('Test subject s' + str(test_subject_id))
    print('Training/Test split:', len(train_x), 'Source Subjects of', train_x[0].shape, test_x[0].shape)
    return train_x, train_y, test_x, test_y
