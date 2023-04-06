# -*- coding: utf-8 -*-
# @Time    : 2021/12/18 11:04
# @Author  : wenzhang
# @File    : dataloader.py
import torch as tr
import numpy as np
from sklearn import preprocessing
from torch.autograd import Variable
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from utils.data_augment import data_aug
from utils.data_utils import traintest_split_cross_subject, time_cut, feature_smooth_moving_average

def data_loader(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'MI1':
        data = np.load('./data/' + dataset + '/MI1.npz')
        X = data['data']
        X = X.reshape(-1, X.shape[2], X.shape[3])
        y = data['label']
        y = y.reshape(-1, )
    elif dataset == 'BNCI2014001-4':
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

    elif dataset == 'MI1':
        paradigm = 'MI'
        num_subjects = 7
        sample_rate = 100
        ch_num = 59
    elif dataset == 'BNCI2014008':
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8

        # time cut
        X = time_cut(X, cut_percentage=0.8)
    elif dataset == 'BNCI2014009':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 16
    elif dataset == 'BNCI2015003':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 8
    elif dataset == 'PhysionetMI':
        paradigm = 'MI'
        num_subjects = 105
        sample_rate = 160
        ch_num = 64

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def data_loader_secondsession(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'MI1':
        data = np.load('./data/' + dataset + '/MI1.npz')
        X = data['data']
        X = X.reshape(-1, X.shape[2], X.shape[3])
        y = data['label']
        y = y.reshape(-1, )
    elif dataset == 'BNCI2014001-4':
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
            #indices.append(np.arange(288) + (576 * i))
            indices.append(np.arange(288) + (576 * i) + 288) # use second sessions
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
            #indices.append(np.arange(100) + (160 * i))
            indices.append(np.arange(60) + (160 * i) + 100) # use second sessions
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
            '''
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))
            '''
            # use second sessions
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

    elif dataset == 'MI1':
        paradigm = 'MI'
        num_subjects = 7
        sample_rate = 100
        ch_num = 59
    elif dataset == 'BNCI2014008':
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8

        # time cut
        X = time_cut(X, cut_percentage=0.8)
    elif dataset == 'BNCI2014009':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 16
    elif dataset == 'BNCI2015003':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 8
    elif dataset == 'PhysionetMI':
        paradigm = 'MI'
        num_subjects = 105
        sample_rate = 160
        ch_num = 64

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def data_loader_feature(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    X = np.load('./data/' + dataset + '_inter_feature_EA\'d.npz')
    y = np.load('./data/' + dataset + '/labels.npy')

    lst = X.files
    X_tmp = []
    for item in lst:
        X_tmp.append(X[item])
    X = np.concatenate(X_tmp, axis=1)

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
                indices.append(np.arange(200) + (400 * (i - 4)) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * (i - 4)) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014008':
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8

        # time cut
        #X = time_cut(X, cut_percentage=0.8)
    elif dataset == 'BNCI2014009':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 16
    elif dataset == 'BNCI2015003':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 8

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def read_mi_all(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    data, label = [], []
    for s in range(args.N):
        # each source sub
        src_data = np.squeeze(Data_raw[s, :, :, :])
        src_label = Label[s, :].reshape(-1, 1)

        if args.aug:
            sample_size = src_data.shape[2]
            # mult_flag, noise_flag, neg_flag, freq_mod_flag
            flag_aug = [True, True, True, True]
            src_data = np.transpose(src_data, (0, 2, 1))
            src_data, src_label = data_aug(src_data, src_label, sample_size, flag_aug)
            src_data = np.transpose(src_data, (0, 2, 1))

        covar = Covariances(estimator=args.cov_type).transform(src_data)
        fea_tsm = TangentSpace().fit_transform(covar)
        src_label = src_label.reshape(-1, 1)

        data.append(fea_tsm)
        label.append(src_label)

    return data, label


def read_mi_train(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # source sub
    src_data = np.squeeze(Data_raw[args.ids, :, :, :])
    src_label = np.squeeze(Label[args.ids, :])
    src_label = tr.from_numpy(src_label).long()
    print(src_data.shape, src_label.shape)  # (288, 22, 750)

    if args.aug:
        sample_size = src_data.shape[2]
        # mult_flag, noise_flag, neg_flag, freq_mod_flag
        flag_aug = [True, True, True, True]
        # flag_aug = [True, False, False, False]
        src_data = np.transpose(src_data, (0, 2, 1))
        src_data, src_label = data_aug(src_data, src_label, sample_size, flag_aug)
        src_data = np.transpose(src_data, (0, 2, 1))
        src_label = tr.from_numpy(src_label).long()
    # print(src_data.shape, src_label.shape)  # (288*7, 22, 750)

    covar = Covariances(estimator=args.cov_type).transform(src_data)
    fea_tsm = TangentSpace().fit_transform(covar)
    fea_tsm = Variable(tr.from_numpy(fea_tsm).float())

    # X.shape - (#samples, # feas)
    print(fea_tsm.shape, src_label.shape)

    return fea_tsm, src_label


def read_mi_test(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # target sub
    tar_data = np.squeeze(Data_raw[args.idt, :, :, :])
    tar_label = np.squeeze(Label[args.idt, :])
    tar_label = tr.from_numpy(tar_label).long()

    # 288 * 22 * 750
    covar_src = Covariances(estimator=args.cov_type).transform(tar_data)
    fea_tsm = TangentSpace().fit_transform(covar_src)

    # covar = Covariances(estimator=cov_type).transform(tar_data)
    # tmp_ref = TangentSpace().fit(covar[:ntu, :, :])
    # fea_tsm = tmp_ref.transform(covar)

    fea_tsm = Variable(tr.from_numpy(fea_tsm).float())

    # X.shape - (#samples, # feas)
    print(fea_tsm.shape, tar_label.shape)
    return fea_tsm, tar_label


def read_mi_test_aug(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # target sub
    tar_data = np.squeeze(Data_raw[args.idt, :, :, :])
    tar_label = np.squeeze(Label[args.idt, :])

    # 288 * 22 * 750
    covar_tar = Covariances(estimator=args.cov_type).transform(tar_data)
    X_tar = TangentSpace().fit_transform(covar_tar)
    X_tar = Variable(tr.from_numpy(X_tar).float())
    y_tar = tr.from_numpy(tar_label).long()

    sample_size = tar_data.shape[2]
    flag_aug = [True, True, True, True]
    tar_data_tmp = np.transpose(tar_data, (0, 2, 1))
    tar_data_tmp, tar_label_aug = data_aug(tar_data_tmp, tar_label, sample_size, flag_aug)
    tar_data_aug = np.transpose(tar_data_tmp, (0, 2, 1))

    # 288 * 22 * 750
    covar_tar = Covariances(estimator=args.cov_type).transform(tar_data_aug)
    X_tar_aug = TangentSpace().fit_transform(covar_tar)
    X_tar_aug = Variable(tr.from_numpy(X_tar_aug).float())
    y_tar_aug = tr.from_numpy(tar_label_aug).long()

    # X.shape - (#samples, # feas)
    print(y_tar.shape, y_tar.shape)
    print(X_tar_aug.shape, y_tar_aug.shape)
    return X_tar, y_tar, X_tar_aug, y_tar_aug


def read_mi_combine(args):  # no data augment
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']
    # print('raw data shape', Data_raw.shape, Label.shape)

    Data_new = Data_raw.copy()
    n_sub = len(Data_raw)

    # MTS transfer
    ids = np.delete(np.arange(0, n_sub), args.idt)
    src_data, src_label = [], []
    for i in range(n_sub - 1):
        src_data.append(np.squeeze(Data_new[ids[i]]))
        src_label.append(np.squeeze(Label[ids[i]]))
    src_data = np.concatenate(src_data, axis=0)
    src_label = np.concatenate(src_label, axis=0)

    # final label
    src_label = np.squeeze(src_label)
    src_label = tr.from_numpy(src_label).long()
    print(src_data.shape, src_label.shape)

    # final features
    covar = Covariances(estimator=args.cov_type).transform(src_data)
    fea_tsm = TangentSpace().fit_transform(covar)
    src_data = Variable(tr.from_numpy(fea_tsm).float())

    return src_data, src_label


def read_mi_combine_tar(args):  # no data augment
    if 'ontinual' in args.method:
        # Continual TTA
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader_secondsession(args.data)
    else:
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(args.data)
    src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def read_features_combine_tar(args):  # no data augment

    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader_feature(args.data)
    src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def data_normalize(fea_de, norm_type):
    if norm_type == 'zscore':
        zscore = preprocessing.StandardScaler()
        fea_de = zscore.fit_transform(fea_de)

    return fea_de


def read_seed_all(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    data, label = [], []
    for s in range(args.N):
        # each source sub
        fea_de = np.squeeze(Data_raw[s, :, :])
        src_label = Label[s, :].reshape(-1, 1)
        data.append(fea_de)
        label.append(src_label)

    return data, label


def read_seed_train(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # source sub
    fea_de = np.squeeze(Data_raw[args.ids, :, :])
    fea_de = data_normalize(fea_de, args.norm)
    fea_de = Variable(tr.from_numpy(fea_de).float())

    src_label = np.squeeze(Label[args.ids, :])
    src_label = tr.from_numpy(src_label).long()
    print(fea_de.shape, src_label.shape)

    return fea_de, src_label


def read_seed_test(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # target sub
    fea_de = np.squeeze(Data_raw[args.idt, :, :])
    fea_de = data_normalize(fea_de, args.norm)
    fea_de = Variable(tr.from_numpy(fea_de).float())

    tar_label = np.squeeze(Label[args.idt, :])
    tar_label = tr.from_numpy(tar_label).long()
    print(fea_de.shape, tar_label.shape)

    return fea_de, tar_label


def read_seed_combine(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']
    print(Data_raw.shape, Label.shape)

    n_sub = len(Data_raw)
    ids = np.delete(np.arange(0, n_sub), args.idt)
    src_data, src_label = [], []
    for i in range(n_sub - 1):
        src_data.append(np.squeeze(Data_raw[ids[i], :, :]))
        src_label.append(np.squeeze(Label[ids[i], :]))

    fea_de = np.concatenate(src_data, axis=0)
    fea_de = data_normalize(fea_de, args.norm)
    fea_de = Variable(tr.from_numpy(fea_de).float())

    src_label = np.concatenate(src_label, axis=0)
    src_label = tr.from_numpy(src_label).long()
    print(fea_de.shape, src_label.shape)

    return fea_de, src_label


def read_seed_combine_tar(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    n_sub = len(Data_raw)
    ids = np.delete(np.arange(0, n_sub), args.idt)
    src_data, src_label = [], []
    for i in range(n_sub - 1):
        src_data.append(np.squeeze(Data_raw[ids[i], :, :]))
        src_label.append(np.squeeze(Label[ids[i], :]))
    src_data = np.concatenate(src_data, axis=0)
    src_label = np.concatenate(src_label, axis=0)

    src_data = data_normalize(src_data, args.norm)
    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = tr.from_numpy(src_label).long()
    print(src_data.shape, src_label.shape)

    # target sub
    tar_data = np.squeeze(Data_raw[args.idt, :, :])
    tar_data = data_normalize(tar_data, args.norm)
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = np.squeeze(Label[args.idt, :])
    tar_label = tr.from_numpy(tar_label).long()
    print(tar_data.shape, tar_label.shape)

    return src_data, src_label, tar_data, tar_label


def obtain_train_val_source(y_array, trial_ins_num, val_type):
    y_array = y_array.numpy()
    ins_num_all = len(y_array)
    src_idx = range(ins_num_all)

    if val_type == 'random':
        # 随机打乱会导致结果偏高，不管是MI还是SEED数据集
        num_train = int(0.9 * len(src_idx))
        id_train, id_val = tr.utils.data.random_split(src_idx, [num_train, len(src_idx) - num_train])

    if val_type == 'last':
        # 按顺序划分，一般情况来说没问题，但是如果源数据类别是按顺序排的，会有问题
        num_train = int(0.9 * trial_ins_num)
        id_train = np.array(src_idx).reshape(-1, trial_ins_num)[:, :num_train].reshape(1, -1).flatten()
        id_val = np.array(src_idx).reshape(-1, trial_ins_num)[:, num_train:].reshape(1, -1).flatten()

    return id_train, id_val
