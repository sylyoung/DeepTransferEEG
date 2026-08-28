import pickle
import numpy as np


def load_data(eeg_path, label_path, args):
    with open(eeg_path, "rb") as f:
        EEG = pickle.load(f)  # list
    with open(label_path, "rb") as f:
        LABEL = pickle.load(f)  # list

    if args.data == 'Driving':
        for k in range(len(EEG)):
            EEG[k] = (EEG[k]).transpose(2, 0, 1)

    return EEG, LABEL


def get_trainset(EEG, LABEL, args):
    testID = args.testID
    train_X = np.concatenate([EEG[i] for i in range(len(EEG)) if i != testID], axis=0)
    train_Y = np.concatenate([LABEL[i] for i in range(len(LABEL)) if i != testID], axis=0)
    
    return train_X, train_Y


def get_testset(EEG, LABEL, args):
    test_X = EEG[args.testID]
    test_Y = LABEL[args.testID]

    return test_X, test_Y
