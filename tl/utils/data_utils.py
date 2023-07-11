import os
import sys

import numpy as np
import scipy.io as sio
import moabb
import mne
import pickle

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2014008, BNCI2014009, BNCI2015003, BNCI2015004, EPFLP300, BNCI2014004, BNCI2015001
from moabb.paradigms import MotorImagery, P300
from scipy.stats import differential_entropy
from scipy.signal import stft
from pykalman import KalmanFilter


sys.path.append("/Users/Riccardo/Workspace/HUST-BCI/repos/EEG/aBCIcomp")



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


def feature_smooth_moving_average(X, ma_window, subject_num, session_num, trial_num, trial_len_arr):
    # (152730, 62, 5) (152730,) 15subjects 3sessions 3394seconds, 62 channels, 5 DE features
    # [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
    trial_len_arr = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206] # in seconds

    def moving_average(X, w):
        if len(X.shape) == 1:
            return np.convolve(X, np.ones(w), 'valid') / w
        elif len(X.shape) == 2:
            out = []
            for i in range(X.shape[1]):
                data = np.convolve(X[:, i], np.ones(w), 'valid') / w
                #print(data.shape)
                out.append(data)
            return np.transpose(np.stack(out, axis=0), (1, 0))
        else:
            print('Moving Average ERROR!')
            return None

    data_smoothed = []
    for subject in range(subject_num):
        for session in range(session_num):
            for trial in range(trial_num):
                chunk_num = trial_len_arr[trial]
                start = subject * (X.shape[0] // (subject_num)) + session * \
                        (X.shape[0] // (subject_num * session_num)) + int(np.sum(trial_len_arr[:trial]))
                #print(start, start+chunk_num)
                pre_ma = X[start:start+chunk_num, :]
                after_ma = moving_average(pre_ma, ma_window)
                data_smoothed.append(after_ma)
    data_smoothed = np.concatenate(data_smoothed, axis=0)
    print('feature smooth before/after:', X.shape, data_smoothed.shape)

    return data_smoothed


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
        # (33600, 8, 257) (33600,) 256Hz 8subjects 1session
    elif dataset_name == 'BNCI2014009':
        dataset = BNCI2014009()
        paradigm = P300()
        # (17280, 16, 206) (17280,) 256Hz 10subjects 3sessions
    elif dataset_name == 'BNCI2015003':
        dataset = BNCI2015003()
        paradigm = P300()
        # (25200, 8, 206) (25200,) 256Hz 10subjects 1session
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


if __name__ == '__main__':
    #dataset_name = 'BNCI2014001'
    #dataset_name = 'BNCI2014002'
    #dataset_name = 'BNCI2014004'
    dataset_name = 'BNCI2015001'
    #dataset_name = 'MI1'
    #dataset_name = 'BNCI2015004'
    #dataset_name = 'BNCI2014008'
    #dataset_name = 'BNCI2014009'
    #dataset_name = 'BNCI2015003'
    #dataset_name = 'EPFLP300'
    info = dataset_to_file(dataset_name, data_save=False)
    print(info)

    #data_folder = '/Users/Riccardo/Workspace/HUST-BCI/data/ERN'
    #dataset_ERN_to_file(data_folder)
    '''
    data_folder = '/Users/Riccardo/Workspace/HUST-BCI/data/SEED/Preprocessed_EEG'
    #data_folder = '/mnt/data2/sylyoung/EEG/SEED/Preprocessed_EEG/'
    dataset_SEED_DE_to_file(data_folder, use_EA=True)
    dataset_SEED_DE_to_file(data_folder, use_EA=False)
    '''
    #data_folder = '/Users/Riccardo/Workspace/HUST-BCI/data/DEAP/data_preprocessed_matlab'
    #data_folder = '/mnt/data2/sylyoung/EEG/SEED/Preprocessed_EEG/'
    #dataset_DEAP_DE_to_file(data_folder, use_EA=True)
    #dataset_DEAP_DE_to_file(data_folder, use_EA=False)

    #data_folder = '/mnt/data2/sylyoung/EEG/SEED-V/SEED-V/EEG_raw/'
    #dataset_SEEDV_DE_to_file(data_folder, use_EA=True)
    #dataset_SEEDV_DE_to_file(data_folder, use_EA=False)

    #data_folder = '/Users/Riccardo/Workspace/HUST-BCI/data/SEED/SEED-V/EEG_DE_features'
    #dataset_SEEDV_extracted_process(data_folder=data_folder)

    #dataset_SEEDV_eye_extracted_process(data_folder='/Users/Riccardo/Workspace/HUST-BCI/data/SEED/SEED-V/Eye_movement_features')

    #feature_smooth_moving_average('/Users/Riccardo/Workspace/HUST-BCI/repos/EEG/data/SEED/X_DE.npy', 5)
    #feature_smooth_moving_average('/Users/Riccardo/Workspace/HUST-BCI/repos/EEG/data/SEED/X_DE_EA.npy')

    #test()

    #dataset_SEEDV_extracted_process('/Users/Riccardo/Workspace/HUST-BCI/data/SEED/SEED-V/EEG_DE_features')
    #dataset_SEEDV_extracted_process('/Users/Riccardo/Workspace/HUST-BCI/data/SEED/SEED-V/Eye_movement_features')


    '''
    BNCI2014001
    <Info | 8 non-empty values
     bads: []
     ch_names: Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, ...
     chs: 22 EEG
     custom_ref_applied: False
     dig: 25 items (3 Cardinal, 22 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 22
     projs: []
     sfreq: 250.0 Hz
    >
    
    BNCI2014002
    <Info | 7 non-empty values
     bads: []
     ch_names: EEG1, EEG2, EEG3, EEG4, EEG5, EEG6, EEG7, EEG8, EEG9, EEG10, ...
     chs: 15 EEG
     custom_ref_applied: False
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 15
     projs: []
     sfreq: 512.0 Hz
    >
    
    BNCI2015004
    <Info | 8 non-empty values
     bads: []
     ch_names: AFz, F7, F3, Fz, F4, F8, FC3, FCz, FC4, T3, C3, Cz, C4, T4, CP3, ...
     chs: 30 EEG
     custom_ref_applied: False
     dig: 33 items (3 Cardinal, 30 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 30
     projs: []
     sfreq: 256.0 Hz
    >
    
    BNCI2014008
    <Info | 8 non-empty values
     bads: []
     ch_names: Fz, Cz, Pz, Oz, P3, P4, PO7, PO8
     chs: 8 EEG
     custom_ref_applied: False
     dig: 11 items (3 Cardinal, 8 EEG)
     highpass: 1.0 Hz
     lowpass: 24.0 Hz
     meas_date: unspecified
     nchan: 8
     projs: []
     sfreq: 256.0 Hz
    >
    
    BNCI2014009
    <Info | 8 non-empty values
     bads: []
     ch_names: Fz, Cz, Pz, Oz, P3, P4, PO7, PO8, F3, F4, FCz, C3, C4, CP3, CPz, CP4
     chs: 16 EEG
     custom_ref_applied: False
     dig: 19 items (3 Cardinal, 16 EEG)
     highpass: 1.0 Hz
     lowpass: 24.0 Hz
     meas_date: unspecified
     nchan: 16
     projs: []
     sfreq: 256.0 Hz
    >
    
    BNCI2015003
    <Info | 8 non-empty values
     bads: []
     ch_names: Fz, Cz, P3, Pz, P4, PO7, Oz, PO8
     chs: 8 EEG
     custom_ref_applied: False
     dig: 11 items (3 Cardinal, 8 EEG)
     highpass: 1.0 Hz
     lowpass: 24.0 Hz
     meas_date: unspecified
     nchan: 8
     projs: []
     sfreq: 256.0 Hz
    >
    '''