import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from scipy.signal import welch, get_window
from getdata import *
from fix_seed import *


def extract_PSD_welch(x, bands=[[4,8],[8,13]], fs=256):
    """
    :param x: n_Epochs * n_Channs * n_Samples
    :param bands: list type bands
    :return:
    """
    print('[Welch] bands: {}'.format(bands))
    N, C = x.shape[:2]
    n_b = len(bands)
    x_fea = np.zeros([N, C*n_b])
    for i in tqdm(range(N)):
        for b in range(n_b):
            psd = []
            lower = bands[b][0]
            upper = bands[b][1]
            for c in range(C):
                fpos, pxx = welch(x[i, c, :], fs=256, window=get_window('hamming', fs * 1))
                mean_psd = np.mean(pxx[(fpos >= lower) & (fpos <= upper)])
                psd.append(mean_psd)
            psd = np.array(psd)
            x_fea[i, b::n_b] = psd
    return x_fea


if __name__ == '__main__':
     os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 4, 5, 6, 7'
     data_name_list = ['Driving', 'Seed']

     for data_name in data_name_list:
          if data_name == 'Driving': 
              paradigm, N, chn, time_sample_num, sample_rate, feature_deep_dim = 'Ecog', 15, 30, 2000, 250, 512
          if data_name == 'New_driving': 
              paradigm, N, chn, time_sample_num, sample_rate, feature_deep_dim = 'Ecog', 27, 30, 750, 250, 512
          if data_name == 'Seed': 
              paradigm, N, chn, time_sample_num, sample_rate, feature_deep_dim = 'Ecog', 23, 17, 1600, 200, 512

          args = argparse.Namespace(feature_deep_dim=feature_deep_dim,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn,  paradigm=paradigm, data=data_name)

          args.method = 'EEGNet'
          args.backbone = 'EEGNet'

          # whether to use EA
          args.align = True
          args.dropout_num = 10

          # cpu or cuda
          args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'

          # get data
          PATH_TO_DATA = "/PATH/TO/DATA/"
          if args.data == 'Driving':
              eeg_path = PATH_TO_DATA + "Driving/Driving_eeg_filter.pkl"
              label_path = PATH_TO_DATA + "Driving/Driving_labels.pkl"
          elif args.data == 'New_driving':
              eeg_path = PATH_TO_DATA + "New_driving/NewDri_eeg.pkl"
              label_path = PATH_TO_DATA + "New_driving/NewDri_label.pkl"
          elif args.data == 'Seed':
              eeg_path = PATH_TO_DATA + "SEED/SEED_eeg_f.pkl"
              label_path = PATH_TO_DATA + "SEED/SEED_labels.pkl"

          EEG, LABEL = load_data(eeg_path, label_path, args)
  
          for testID in range(N):
               args.testID = testID
               tar_data, tar_label = get_testset(EEG, LABEL, args)
 
               tar_data = torch.tensor(tar_data, dtype=torch.float32)
               tar_label = torch.tensor(tar_label, dtype=torch.float32)
               print(tar_data.shape)
 
               SEED = 42
               fix_random_seed(SEED)
               
               PATH_TO_PKL = "/PATH/TO/PKL/"
               psd_feature = extract_PSD_welch(tar_data)
               print(psd_feature.shape)
               save_path = PATH_TO_PKL + data_name + '/s' + str(args.testID)
               if os.path.isdir(save_path):
                    pass
               else:
                    os.makedirs(save_path)
               torch.save({'data': psd_feature, 'labels': tar_label}, save_path + '/dataset.pt')
