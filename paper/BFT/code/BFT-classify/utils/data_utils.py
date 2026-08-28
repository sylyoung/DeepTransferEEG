import numpy as np
from sklearn import preprocessing


def data_process(dataset):
    '''
    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    DATA_PATH = "/PATH/TO/SAVE/DATA/"
    X = np.load(DATA_PATH + dataset + '/X.npy')
    y = np.load(DATA_PATH + dataset + '/labels.npy')
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

    elif dataset == 'Schirrmeister2017':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 500
        ch_num = 128
        trials_arr = np.array([[320, 0],
                    [813, 0],
                    [880, 0],
                    [897, 0],
                    [720, 0],
                    [880, 0],
                    [880, 0],
                    [654, 0],
                    [880, 0],
                    [880, 0],
                    [880, 0],
                    [880, 0],
                    [800, 0],
                    [880, 0]])
        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(trials_arr[i, 0]) + np.sum(trials_arr[:i, :]))
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

    elif dataset == 'Zhou2016':
        paradigm = 'MI'
        num_subjects = 4
        sample_rate = 250
        ch_num = 14
        trials_arr = np.array([[179, 150, 150],
                                [150, 135, 150],
                                [150, 151, 150],
                                [135, 150, 150]])
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(trials_arr[i, 0]) + np.sum(trials_arr[:i, :]))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def split_data_by_subject(X, y, trails_num):
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")
    if sum(trails_num) != len(X):
        raise ValueError("The sum of trails_num must equal the number of samples in X.")

    data_subjects = np.split(X, np.cumsum(trails_num)[:-1], axis=0)
    labels_subjects = np.split(y, np.cumsum(trails_num)[:-1], axis=0)

    return data_subjects, labels_subjects


def get_test_train(data_subjects, labels_subjects, idt):
    test_x = data_subjects[idt]
    test_y = labels_subjects[idt]
    
    train_x = np.concatenate([data_subjects[i] for i in range(len(data_subjects)) if i != idt], axis=0)
    train_y = np.concatenate([labels_subjects[i] for i in range(len(data_subjects)) if i != idt], axis=0)
    
    print('Test subject:', idt)
    print('Training/Test split:', train_x.shape, test_x.shape)
    
    return train_x, train_y, test_x, test_y