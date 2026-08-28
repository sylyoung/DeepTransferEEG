import os 
import sys
import torch
import torch.nn as nn
from models.EEGNet import *
from models.losspredictor import *

import argparse
from utils.EA import *
from dropout import *
from augment import *
from utils.data_utils import *

# the corruption bank of Section IV-F lives one directory up, so that the
# classification and the regression experiments share the same definitions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from corruptions import apply_artifact


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6, 7'
    data_name_list = ['Zhou2016', 'Schirrmeister2017']

    for data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        if data_name == 'BNCI2014001': 
            paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = \
            'MI', 9, 22, 2, 1001, 250, [144, 144, 144, 144, 144, 144, 144, 144, 144]
        if data_name == 'Zhou2016': 
            paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = \
            'MI', 4, 14, 2, 1251, 250, [119, 100, 100, 90]
        if data_name == 'Schirrmeister2017': 
            paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = \
            'MI', 14, 128, 2, 2001, 500, [160, 406, 440, 448, 360, 440, 440, 327, 441, 440, 440, 440, 400, 440]

        args = argparse.Namespace(trial_num=trial_num, time_sample_num=time_sample_num, 
                                  sample_rate=sample_rate, N=N, chn=chn, 
                                  class_num=class_num, paradigm=paradigm, data=data_name)

        args.method = 'EEGNet'
        args.backbone = 'EEGNet'

        # whether to use EA
        args.align = True
        args.dropout_num = 10

        # Section IV-F, Fig. 5: the test-time corruption applied to the input
        # signal. 'clean' is the uncorrupted condition of Tables II and III;
        # for Fig. 5 set it to one of corruptions.ARTIFACT_NAMES, of which
        # 'temporal_segment_noise' and 'channel_noise' are the temporal and the
        # spatial Gaussian noise the figure reports, at severity 1, 2 or 3.
        args.corruption = 'clean'
        args.severity = 2
        args.corruption_seed = 42

        # cpu or cuda
        args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'

        # load data
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)
        data_subjects, labels_subjects = split_data_by_subject(X, y, args.trial_num)
        
        # if the data has not undergone EA, then perform EA.
        # if the data has already been saved after EA, just load the data directly and skip this step.
        # the corruption is injected before EA, so that the whole test-time
        # pipeline, alignment included, sees the degraded signal. Every subject
        # is corrupted because every subject is the target of one fold, and the
        # source arrays this script loads are not used at test time.
        for i in range(len(data_subjects)):
            if args.corruption != 'clean':
                data_subjects[i] = apply_artifact(data_subjects[i], args.corruption,
                                                  args.severity, args.sample_rate,
                                                  args.corruption_seed + i)
            data_subjects[i] = EA_online(data_subjects[i])

        for SEED in [42, 43, 44]:
            args.SEED = SEED
            fix_random_seed(SEED)
            for idt in range(N):
                # target subject
                args.idt = idt
                src_data, src_label, tar_data, tar_label = get_test_train(data_subjects, labels_subjects, idt)
                
                eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
                model_target = EEGNet(n_classes=args.class_num,
                                Chans=args.chn,
                                Samples=eeg_length,
                                kernLength=int(args.sample_rate // 2),
                                F1=8,
                                D=2,
                                F2=16,
                                dropoutRate=0.25)   
                PATH_EEGNET_MODEL = "/PATH/TO/SAVE/MODEL/"
                PATH_TO_LOSSPRE_MODEL_DROPOUT = "/PATH/TO/SAVE/MODEL/"  

                base_dir = PATH_EEGNET_MODEL + str(SEED) + '/EEGNet_pth/'
                tar_model_dir = base_dir + args.data + '/s' + str(args.idt) + '/EEGNet_epoch_200.pth'
                tar_model_dir_cc = tar_model_dir
                checkpoint = torch.load(tar_model_dir)
                model_target.load_state_dict(checkpoint)
                model_target = model_target.cuda()

                block_model = EEGNet_Block(model_target.block1, model_target.block2)
                classifier = EEGNet_Classifier(model_target.classifier_block)
                block_model = block_model.cuda()
                classifier = classifier.cuda()
                
                # get accurancy of different transformation
                test_augment(model_target, tar_data, tar_label, args)
                # get accurancy of BN-adapt
                test_BNadapt(model_target, tar_data, tar_label, args)
                checkpoint = torch.load(tar_model_dir_cc)
                model_target.load_state_dict(checkpoint)
                model_target = model_target.cuda()

                block_model = EEGNet_Block(model_target.block1, model_target.block2)
                classifier = EEGNet_Classifier(model_target.classifier_block)
                block_model = block_model.cuda()
                classifier = classifier.cuda()

                model_loss = EEGNetLossPredictor(F2=16, Samples=eeg_length) 
                base_dir = PATH_TO_LOSSPRE_MODEL_DROPOUT + str(SEED) + '/loss_model_dropout_new(batch=16)/'
                loss_model_dir = base_dir + args.data + '/s' + str(args.idt) + '/loss_pre/EEGNetLossPredictor_epoch_20.pth'
    
                checkpoint = torch.load(loss_model_dir)
                model_loss.load_state_dict(checkpoint)
                model_loss = model_loss.cuda()

                # get accurancy of BFT-A
                test_augment_with_loss(model_loss, model_target, block_model, tar_data, tar_label, args)
                checkpoint = torch.load(tar_model_dir_cc)
                model_target.load_state_dict(checkpoint)
                model_target = model_target.cuda()

                block_model = EEGNet_Block(model_target.block1, model_target.block2)
                classifier = EEGNet_Classifier(model_target.classifier_block)
                block_model = block_model.cuda()
                classifier = classifier.cuda()
                # get accurancy of MC Dropout
                test_dropout(block_model, classifier, tar_data, tar_label, args)
                # get accurancy of BFT-D
                test_dropout_with_loss(model_loss, block_model, classifier, tar_data, tar_label, args)
