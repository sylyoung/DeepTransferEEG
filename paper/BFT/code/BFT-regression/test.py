import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import sys
import argparse
from utils.EA import *
from utils.getdata import *
from utils.fix_seed import *

from models.Deformer import *
from models.EEGNet import *
from models.lossPredictor import *
from augment_utils import *
from dropout_utils import *

# the corruption bank of Section IV-F lives one directory up, so that the
# classification and the regression experiments share the same definitions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from corruptions import apply_artifact


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'
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

        # Section IV-F, Fig. 6: the test-time corruption applied to the input
        # signal. 'clean' is the uncorrupted condition of Table V; for Fig. 6
        # set it to one of corruptions.ARTIFACT_NAMES, of which
        # 'temporal_segment_noise' and 'channel_noise' are the temporal and the
        # spatial Gaussian noise the figure reports, at severity 1, 2 or 3.
        args.corruption = 'clean'
        args.severity = 2
        args.corruption_seed = 42

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

        # if the data has not undergone EA, then perform EA.
        # if the data has already been saved after EA, just load the data directly and skip this step.
        # the corruption is injected before EA, so that the whole test-time
        # pipeline, alignment included, sees the degraded signal. Every subject
        # is corrupted because every subject is the target of one fold.
        for i in range(len(EEG)):
            if args.corruption != 'clean':
                EEG[i] = apply_artifact(EEG[i], args.corruption, args.severity,
                                        args.sample_rate, args.corruption_seed + i)
            EEG[i] = EA_online(EEG[i])

        for SEED in [42, 43, 44]:
            args.SEED = SEED
            fix_random_seed(SEED)
            for testID in range(N):
                args.testID = testID
                tar_data, tar_label = get_testset(EEG, LABEL, args)
                print(args.data, '  s' + str(testID))
                print("Target Data Shape: ", tar_data.shape, "Target Label Shape: ", tar_data.shape)

                eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
                EEGNet_model = EEGNet(Chans=args.chn,
                                Samples=eeg_length,
                                kernLength=int(args.sample_rate // 2),
                                F1=8,
                                D=2,
                                F2=16,
                                dropoutRate=0.25)  
                base_model = EEGNet_Block(EEGNet_model.block1, EEGNet_model.block2)
                regression_model = EEGNet_Regression(EEGNet_model.regression_block) 

                PATH_TO_MODEL = '/PATH/TO/SAVE/MODEL/'
                PATH_TO_LOSSPRE_MODEL_DROPOUT = '/PATH/TO/SAVE/MODEL/'
                base_dir = PATH_TO_MODEL + str(SEED) + '/EEGNet/New_EEGNetBlock/'
                tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/EEGNetBlock_epoch_100.pth'
                tar_model_dir_cc = tar_model_dir
                checkpoint = torch.load(tar_model_dir)
                base_model.load_state_dict(checkpoint)
                base_model = base_model.cuda()

                base_dir = PATH_TO_MODEL + str(SEED) + '/EEGNet/New_Regression_head/'
                tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/Regression_head_epoch_100.pth'
                checkpoint = torch.load(tar_model_dir)
                regression_model.load_state_dict(checkpoint)
                regression_model = regression_model.cuda()

                # get the results of different transformation
                test_augment(base_model, regression_model, tar_data, tar_label, args)
                # get the results of BN-adapt
                test_BNadapt(base_model, regression_model, tar_data, tar_label, args)
                checkpoint = torch.load(tar_model_dir_cc)
                base_model.load_state_dict(checkpoint)
                base_model = base_model.cuda()

                F2 = 16
                input_dim = F2 * (eeg_length // (4 * 8))
                model_loss = LossPredictor(input_dim)

                base_dir = PATH_TO_LOSSPRE_MODEL_DROPOUT + str(SEED) + '/EEGNet/New_loss_model_dropout/'
                loss_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/LossPredictor_epoch_20.pth'

                checkpoint = torch.load(loss_model_dir)
                model_loss.load_state_dict(checkpoint)
                model_loss = model_loss.cuda()

                # get the results of BFT-A
                test_augment_with_loss(model_loss, base_model, regression_model, tar_data, tar_label, args)

                checkpoint = torch.load(tar_model_dir_cc)
                base_model.load_state_dict(checkpoint)
                base_model = base_model.cuda()
                # get the results of MC dropout
                test_dropout(base_model, regression_model, tar_data, tar_label, args)
                # get the results of BFT-D
                test_dropout_with_loss(model_loss, base_model, regression_model, tar_data, tar_label, args)
