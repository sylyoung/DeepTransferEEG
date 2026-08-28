import torch
import torch.nn as nn
from models.EEGNet import *
from models.losspredictor import *

import os
import argparse
from utils.EA import *
from utils.data_utils import *


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

        # learning rate
        if data_name == 'BNCI2014001':
            args.lr = 0.001
        if data_name == 'BNCI2014004':
            args.lr = 0.001
        if data_name == 'Zhou2016':
            args.lr = 0.001
        if data_name == 'Schirrmeister2017':
            args.lr = 0.001

        # train batch size
        args.batch_size = 16

        if data_name == 'BNCI2014001':
            args.max_epoch = 20
        if data_name == 'BNCI2014004':
            args.max_epoch = 40
        if data_name == 'Zhou2016':
            args.max_epoch = 40
        if data_name == 'Schirrmeister2017':
            args.max_epoch = 20

        # cpu or cuda
        args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'

        # load data
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)
        data_subjects, labels_subjects = split_data_by_subject(X, y, args.trial_num)
        
        # EA
        for i in range(len(data_subjects)):
            data_subjects[i] = EA_offline(data_subjects[i], 1)

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
                base_dir = PATH_EEGNET_MODEL + str(SEED) + '/EEGNet_pth/'
                tar_model_dir = base_dir + args.data + '/s' + str(args.idt) + '/EEGNet_epoch_200.pth'
                checkpoint = torch.load(tar_model_dir)
                model_target.load_state_dict(checkpoint)
                model_target = model_target.cuda()

                block_model = EEGNet_Block(model_target.block1, model_target.block2)
                model_loss = EEGNetLossPredictor(F2=16, Samples=eeg_length)
                classifier = EEGNet_Classifier(model_target.classifier_block)
                block_model = block_model.cuda()
                classifier = classifier.cuda()
                model_loss = model_loss.cuda()

                # learn_augment_loss(model_loss, model_target, block_model, src_data, src_label, args)
                learn_dropout_loss(model_loss, block_model, classifier, src_data, src_label, args)
