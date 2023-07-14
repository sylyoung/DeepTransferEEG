import mne
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import random
import sys
import os

from utils.alg_utils import EA
from tl.ttime_ensemble import SML_soft, SML_soft_multiclass


def apply_zscore(train_x, test_x, num_subjects):
    # train split into subjects
    train_z = []
    trial_num = int(train_x.shape[0] / (num_subjects - 1))
    for j in range(num_subjects - 1):
        scaler = preprocessing.StandardScaler()
        train_x_tmp = scaler.fit_transform(train_x[trial_num * j: trial_num * (j + 1), :])
        train_z.append(train_x_tmp)
    train_x = np.concatenate(train_z, axis=0)
    # test subject
    scaler = preprocessing.StandardScaler()
    test_x = scaler.fit_transform(test_x)
    return train_x, test_x


def data_loader(dataset):
    '''
    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''
    mne.set_log_level('warning')

    if dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))
        indices = np.concatenate(indices, axis=0)

        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def data_alignment(X, num_subjects):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X


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


def ml_classifier(approach, output_probability, train_x, train_y, test_x, return_model=None, weight=None):
    if approach == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif approach == 'LR':
        clf = LogisticRegression(max_iter=1000)
    elif approach == 'AdaBoost':
        clf = AdaBoostClassifier()
    elif approach == 'GradientBoosting':
        clf = GradientBoostingClassifier()
    elif approach == 'xgb':
        clf = XGBClassifier()
        if weight:
            print('XGB weight:', weight)
            clf = XGBClassifier(scale_pos_weight=weight)
            # clf = imb_xgb(special_objective='focal', focal_gamma=2.0)
    # clf = LinearDiscriminantAnalysis()
    # clf = SVC()
    # clf = LinearSVC()
    # clf = KNeighborsClassifier()
    clf.fit(train_x, train_y)

    if output_probability:
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    if return_model:
        return pred, clf
    else:
        print(pred)
        return pred


def ml_multisource(dataset, info, align, approach, cuda_device_id):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(dataset)
    print('X, y, num_subjects, paradigm, sample_rate:', X.shape, y.shape, num_subjects, paradigm, sample_rate)

    print('sample rate:', sample_rate)

    if align:
        X = data_alignment(X, num_subjects)

    scores_arr = []

    for i in range(num_subjects):
        train_x, train_y, test_x, test_y = traintest_split_multisource(dataset, X, y, num_subjects, i)
        print('num of train_x, train_x, train_y, test_x, test_y.shape', len(train_x), train_x[0].shape, train_y[0].shape, test_x.shape, test_y.shape)

        if paradigm == 'MI':
            # CSP
            subj_scores = []
            subj_preds = []
            for s in range(len(train_x)):
                subj_train_x, subj_train_y = train_x[s], train_y[s]
                subj_csp = mne.decoding.CSP(n_components=10)
                subj_train_x_csp = subj_csp.fit_transform(subj_train_x, subj_train_y)
                subj_test_x_csp = subj_csp.transform(test_x)

                # classifier
                subj_pred, subj_model = ml_classifier(approach, True, subj_train_x_csp, subj_train_y, subj_test_x_csp, return_model=True)
                subj_preds.append(subj_pred)
            subj_preds = np.stack(subj_preds)

            # SML
            if dataset == 'BNCI2014001-4':
                pred = SML_soft_multiclass(subj_preds[:, :])
            else:
                pred = SML_soft(subj_preds[:, :, 1])

            # averaging
            #avg_pred = np.average(subj_preds, axis=0)
            #pred = np.argmax(avg_pred, axis=1)

            subj_scores = np.round(accuracy_score(test_y, pred), 5)
            score = np.mean(subj_scores)
        scores_arr.append(score)
    print('#' * 30)
    for i in range(len(scores_arr)):
        scores_arr[i] = np.round(scores_arr[i] * 100)
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))

    return scores_arr


if __name__ == '__main__':

    # cuda_device_id as args[1]
    if len(sys.argv) > 1:
        cuda_device_id = str(sys.argv[1])
    else:
        cuda_device_id = -1
    try:
        device = torch.device('cuda:' + cuda_device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_id
        print('using GPU')
    except:
        device = torch.device('cpu')
        print('using CPU (no CUDA)')

    scores = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_arr = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014001-4']

    for dataset in dataset_arr:

        for approach in ['LDA']:
            # use EA
            align = True

            print(dataset, align, approach)

            # info = dataset_to_file(dataset, data_save=False)

            ml_multisource(dataset, None, align, approach, cuda_device_id)
