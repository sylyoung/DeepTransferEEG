import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import argparse
from utils.EA import *
from utils.getdata import *
from utils.fix_seed import *
from utils.augment import *


from models.Deformer import *
from models.EEGNet import *


def train_regression_model(base_model, regression_model, X_train, labels_train, args):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.float32)  
    labels_train = labels_train.squeeze()

    data_train = torch.utils.data.TensorDataset(X_train, labels_train)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    criterion = nn.MSELoss()  
    optimizer = optim.Adam(list(base_model.parameters()) + list(regression_model.parameters()), lr=args.lr)

    max_iter = args.max_epoch * len(loader_train)
    # max(1, ...): the unguarded expression is 0 for any max_epoch below 10,
    # and iter_num % 0 raises ZeroDivisionError on the first iteration
    interval_iter = max(1, int(args.max_epoch / 10) * max_iter // args.max_epoch)
    iter_num = 0
    epoch_loss = 0
    cnt = 0

    base_model.train()
    regression_model.train()
    while iter_num < max_iter:
        try:
            inputs_train, labels_train = next(iter_train)
        except:
            iter_train = iter(loader_train)
            inputs_train, labels_train = next(iter_train)

        if inputs_train.size(0) == 1:
            continue

        iter_num += 1
        inputs_train = inputs_train.detach().cpu().numpy()
        labels_train = labels_train.detach().cpu().numpy()
        inputs_train, labels_train = random_aug(inputs_train, labels_train, args)
        inputs_train, labels_train = inputs_train.cuda(), labels_train.cuda()

        hideen_features = base_model(inputs_train)
        outputs_train = regression_model(hideen_features).squeeze()

        regression_loss = criterion(outputs_train, labels_train)

        epoch_loss += regression_loss.item()
        cnt += 1

        optimizer.zero_grad()
        regression_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_model.eval()
            regression_model.eval()

            epoch_loss_avg = epoch_loss / cnt
            print('Epoch:{}/{}; Epoch Loss = {:.2f}'
                  .format(int(iter_num // len(loader_train)), 
                          int(max_iter // len(loader_train)), 
                          epoch_loss_avg))

            PATH_TO_MODEL = '/PATH/TO/MODEL/'
            CHECKPOINT_DIR = PATH_TO_MODEL + str(args.SEED) + "/EEGNet/New_EEGNetBlock/"
            path = CHECKPOINT_DIR + args.data + "/s" + str(args.testID)
            os.makedirs(path, exist_ok=True)
            torch.save(base_model.state_dict(), path + "/EEGNetBlock_epoch_" + str(iter_num) + ".pth")
            
            CHECKPOINT_DIR = PATH_TO_MODEL + str(args.SEED) + "/EEGNet/New_Regression_head/"
            path = CHECKPOINT_DIR + args.data + "/s" + str(args.testID)
            os.makedirs(path, exist_ok=True)
            torch.save(regression_model.state_dict(), path + "/Regression_head_epoch_" + str(iter_num) + ".pth")

            base_model.train()
            regression_model.train()
            epoch_loss = 0
            cnt = 0


def train_regression_model_noaug(base_model, regression_model, X_train, labels_train, args):
    eeg_length = (round(args.time_sample_num / args.sample_rate) - 1) * args.sample_rate
    X_train = X_train[:, :, :eeg_length]
    X_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.float32)  
    X_train = X_train.unsqueeze(1)
    labels_train = labels_train.squeeze()

    data_train = torch.utils.data.TensorDataset(X_train, labels_train)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    criterion = nn.MSELoss()  
    optimizer = optim.Adam(list(base_model.parameters()) + list(regression_model.parameters()), lr=args.lr)

    max_iter = args.max_epoch * len(loader_train)
    # max(1, ...): the unguarded expression is 0 for any max_epoch below 10,
    # and iter_num % 0 raises ZeroDivisionError on the first iteration
    interval_iter = max(1, int(args.max_epoch / 10) * max_iter // args.max_epoch)
    iter_num = 0
    epoch_loss = 0
    cnt = 0

    base_model.train()
    regression_model.train()
    while iter_num < max_iter:
        try:
            inputs_train, labels_train = next(iter_train)
        except:
            iter_train = iter(loader_train)
            inputs_train, labels_train = next(iter_train)

        if inputs_train.size(0) == 1:
            continue

        iter_num += 1
        inputs_train, labels_train = inputs_train.cuda(), labels_train.cuda()

        hideen_features = base_model(inputs_train)
        outputs_train = regression_model(hideen_features).squeeze()

        regression_loss = criterion(outputs_train, labels_train)

        epoch_loss += regression_loss.item()
        cnt += 1

        optimizer.zero_grad()
        regression_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_model.eval()
            regression_model.eval()

            epoch_loss_avg = epoch_loss / cnt
            print('Epoch:{}/{}; Epoch Loss = {:.2f}'
                  .format(int(iter_num // len(loader_train)), 
                          int(max_iter // len(loader_train)), 
                          epoch_loss_avg))

            PATH_TO_MODEL_NO_AUG = '/PATH/TO/MODEL/'
            CHECKPOINT_DIR = PATH_TO_MODEL_NO_AUG + str(args.SEED) + "/EEGNet/New_EEGNetBlock_noaug/"
            path = CHECKPOINT_DIR + args.data + "/s" + str(args.testID)
            os.makedirs(path, exist_ok=True)
            torch.save(base_model.state_dict(), path + "/EEGNetBlock_epoch_" + str(iter_num) + ".pth")
            
            CHECKPOINT_DIR = PATH_TO_MODEL_NO_AUG + str(args.SEED) + "/EEGNet/New_Regression_head_noaug/"
            path = CHECKPOINT_DIR + args.data + "/s" + str(args.testID)
            os.makedirs(path, exist_ok=True)
            torch.save(regression_model.state_dict(), path + "/Regression_head_epoch_" + str(iter_num) + ".pth")

            base_model.train()
            regression_model.train()
            epoch_loss = 0
            cnt = 0


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4, 5, 6, 7'
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

        # learning rate
        if data_name == 'Driving':
            args.lr = 0.001
        if data_name == 'New_driving':
            args.lr = 0.001
        if data_name =='Seed':
            args.lr = 0.001

        # train batch size
        args.batch_size = 64
        
        if data_name == 'Driving':
            args.max_epoch = 100
        if data_name == 'New_driving':
            args.max_epoch = 100
        if data_name =='Seed':
            args.max_epoch = 100

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

        for i in range(len(EEG)):
            EEG[i] = EA_offline(EEG[i], 1)

        for SEED in [42, 43, 44]:
            args.SEED = SEED
            fix_random_seed(SEED)
            for testID in range(N):
                # target subject
                args.testID = testID
                src_data, src_label = get_trainset(EEG, LABEL, args)
                print("Source Data Shape: ", src_data.shape, "Source Label Shape: ", src_label.shape)

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
                base_model.cuda()
                regression_model.cuda()

                train_regression_model(base_model, regression_model, src_data, src_label, args)
                # train_regression_model_noaug(base_model, regression_model, src_data, src_label, args)
            
