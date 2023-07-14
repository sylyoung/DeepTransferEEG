import os

import numpy as np
import moabb

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2015001
from moabb.paradigms import MotorImagery, P300


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
    elif dataset_name == 'BNCI2015001':
        dataset = BNCI2015001()
        paradigm = MotorImagery(n_classes=2)
        # (5600, 13, 2561) (5600,) 512Hz 12subjects * 2 classes * (200 + 200 + (200 for Subj 8/9/10/11)) trials * (2/3)sessions

    if data_save:
        print('preparing ' + str(dataset_name) + ' data...')
        if dataset_name == 'PhysionetMI':
            X, labels, meta = paradigm.get_data(dataset=dataset,
                                                subjects=list(np.delete(dataset.subject_list, [87, 91, 99, 103])))
        else:
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:])
        ar_unique, cnts = np.unique(labels, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts)
        print(X.shape, labels.shape)
        if not os.path.exists('./data/'):
            os.makedirs('./data/')
        if not os.path.exists('./data/' + dataset_name + '/'):
            os.makedirs('./data/' + dataset_name + '/')
        np.save('./data/' + dataset_name + '/X', X)
        np.save('./data/' + dataset_name + '/labels', labels)
        meta.to_csv('./data/' + dataset_name + '/meta.csv')
        print('done!')
    else:
        if isinstance(paradigm, MotorImagery):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            return X.info
        elif isinstance(paradigm, P300):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            return X.info


if __name__ == '__main__':

    datasets = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']
    for dataset_name in datasets:
        info = dataset_to_file(dataset_name, data_save=True)

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

    BNCI2015001
    <Info | 8 non-empty values
     bads: []
     ch_names: FC5, FC3, FC1, FCz, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, ...
     chs: 64 EEG
     custom_ref_applied: False
     dig: 67 items (3 Cardinal, 64 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: 2009-08-12 16:15:00 UTC
     nchan: 64
     projs: []
     sfreq: 160.0 Hz
    >
    '''