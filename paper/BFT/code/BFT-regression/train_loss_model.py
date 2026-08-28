# Training of the ranking module r for the regression tasks, the "Ranking
# Module Training" block of Algorithm 1.  Equation numbers refer to the paper
# "Backpropagation-Free Test-Time Adaptation for Lightweight EEG-Based BCIs"
# (IEEE J. Biomed. Health Inform., 2026):
#
#   Eq. (5)  w_{i,k} = softmax_k( r(z_i^(k)) )
#   Eq. (6)  L_ranking = || m(w_i) - pi_i ||_1
#
# The task loss used to build the target profile pi_i is the MSE of the
# regression head, in place of the cross entropy used for classification.
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import argparse
from utils.EA import *
from utils.getdata import *
from utils.fix_seed import *
from utils.load_dataAuged import *

from models.Deformer import *
from models.EEGNet import *
from models.lossPredictor import *

import sys
PATH_TO_SODEEP = '../sodeep'
sys.path.append(PATH_TO_SODEEP)
from sodeep import load_sorter, SpearmanLoss


def learn_dropout_loss(model_loss, base_model, regression_model, X_train, labels_train, args):
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_train = X_train[:, :, :eeg_length]
    X_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.float32)
    X_train = X_train.unsqueeze(1)
    labels_train = labels_train.squeeze()
    X_train, labels_train = X_train.cuda(), labels_train.cuda()

    data_train = torch.utils.data.TensorDataset(X_train, labels_train)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, 
                                               shuffle=True, drop_last=True)
    
    loss_fn = nn.MSELoss()
    # Eq. (6).  NOTE: SpearmanLoss defaults to lbd = 0, at which its forward
    # reduces to a plain L1 between the predicted and target Softmax profiles
    # and the mapping module m contributes no gradient.  Pass lbd=1 to use the
    # rank space that Eq. (6) describes.
    sorter_checkpoint_path = PATH_TO_SODEEP + '/weights/10th_100epochs_best_model.pth.tar'
    criterion = SpearmanLoss(*load_sorter(sorter_checkpoint_path))
    criterion.cuda()
    optimizer = optim.Adam(model_loss.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(loader_train)
    # max(1, ...): the unguarded expression is 0 for any max_epoch below 10,
    # and iter_num % 0 raises ZeroDivisionError on the first iteration
    interval_iter = max(1, int(args.max_epoch / 10) * max_iter // args.max_epoch)
    iter_num = 0
    epoch_loss = 0
    cnt = 0

    base_model.eval()
    regression_model.eval()
    model_loss.train()

    num_splits = args.dropout_num
    drop_ranges = [(i / num_splits, (i + 1) / num_splits) for i in range(num_splits)]
    range_keys = [f"{start:.1f}-{end:.1f}" for start, end in drop_ranges]

    while iter_num < max_iter:
        try:
            inputs_train, labels_train = next(iter_train)
        except:
            iter_train = iter(loader_train)
            inputs_train, labels_train = next(iter_train)

        if inputs_train.size(0) == 1:
            continue

        iter_num += 1

        pred_losses = []
        dropout_loss_list = []
        output1 = base_model(inputs_train)
        B, D = output1.shape
        for (start_r, end_r), key in zip(drop_ranges, range_keys):
            output1_mask = output1.clone()
            start = int(start_r * D)
            end = int(end_r * D)
            output1_mask[:, start:end] = 0.0

            this_outputs = (regression_model(output1_mask)).squeeze()
            this_outputs = this_outputs / (1 - 1 / args.dropout_num)
            this_loss = loss_fn(this_outputs, labels_train)
            dropout_loss_list.append(this_loss.item())

            pred_losses.append(model_loss(output1_mask))
        # dim = 10
        real_losses = torch.tensor(dropout_loss_list)
        pred_losses = torch.stack(pred_losses).squeeze()
        pred_losses = pred_losses.mean(dim=1)   
        relative_real_losses = F.softmax(-real_losses, dim=0)
        predicted_probs = F.softmax(-pred_losses, dim=0)
        relative_real_losses = relative_real_losses.cuda()

        loss = criterion(predicted_probs, relative_real_losses)
        epoch_loss += loss.item()
        cnt += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            epoch_loss_avg = epoch_loss / cnt
            print('Epoch:{}/{}; Epoch Loss = {:.5f}'
                  .format(int(iter_num // len(loader_train)), 
                          int(max_iter // len(loader_train)), 
                          epoch_loss_avg))

            PATH_TO_LOSSPRE_MODEL_DROPOUT = '/PATH/TO/SAVE/MODEL/'
            CHECKPOINT_DIR = PATH_TO_LOSSPRE_MODEL_DROPOUT + str(args.SEED) + "/EEGNet/New_loss_model_dropout/"
            path = CHECKPOINT_DIR + args.data + "/s" + str(args.testID)
            os.makedirs(path, exist_ok=True)
            torch.save(model_loss.state_dict(), path + "/LossPredictor_epoch_" + str(iter_num) + ".pth")


def compute_real_losses(augmented_inputs, base_model, regression_model):
    loss_fn = nn.MSELoss()
    real_losses = []
    base_model.eval()
    regression_model.eval()
    with torch.no_grad():
        for x_aug, label in augmented_inputs:
            label = label.squeeze()
            pred = regression_model(base_model(x_aug)).squeeze()
            loss = loss_fn(pred, label)
            real_losses.append(loss.item())
    return torch.tensor(real_losses)  # dim = 12


def learn_augment_loss(model_loss, base_model, regression_model, X_train, labels_train, args):
    aug_names = [
        "identity", "noise",
        "mult_0.9", "mult_1.1", "mult_1.2",
        "freq_high", "freq_low",
        "slide_1", "slide_2", "slide_3", "slide_4", "slide_5"
    ]
    PATH_TO_AUGED_DATA = '/PATH/TO/AUGED/DATA/'
    data_root = PATH_TO_AUGED_DATA + args.data

    test_id = args.testID
    all_ids = list(range(args.N))
    subject_ids = [i for i in all_ids if i != test_id] 

    aug_dataset = AugmentedDataset(data_root, subject_ids, aug_names)
    loader_train = torch.utils.data.DataLoader(
        aug_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Eq. (6), same lbd = 0 caveat as above.
    # NOTE: the path separator is missing, so this resolves to
    # '../sodeepweights/...' and the checkpoint cannot be found as written.
    sorter_checkpoint_path = PATH_TO_SODEEP + '/weights/12th_100epochs_best_model.pth.tar'
    criterion = SpearmanLoss(*load_sorter(sorter_checkpoint_path))
    criterion.cuda()
    optimizer = optim.Adam(model_loss.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(loader_train)
    # max(1, ...): the unguarded expression is 0 for any max_epoch below 10,
    # and iter_num % 0 raises ZeroDivisionError on the first iteration
    interval_iter = max(1, int(args.max_epoch / 10) * max_iter // args.max_epoch)
    iter_num = 0
    epoch_loss = 0
    cnt = 0

    base_model.eval()
    regression_model.eval()
    model_loss.train()
    while iter_num < max_iter:
        try:
            inputs_train = next(iter_train)
        except:
            iter_train = iter(loader_train)
            inputs_train = next(iter_train)

        if inputs_train[0][0].size(0) == 1:
            continue
        
        iter_num += 1
        trans_traininputs = []
        for i in range(len(inputs_train)):
            trans_traininputs.append((inputs_train[i][0].cuda(), inputs_train[i][1].cuda()))
        
        x_aug_list = trans_traininputs

        real_losses = compute_real_losses(x_aug_list, base_model, regression_model)
        relative_real_losses = F.softmax(-real_losses, dim=0)
        relative_real_losses = relative_real_losses.cuda()

        pred_losses = []
        for i in range(len(x_aug_list)):
            x, _ = x_aug_list[i]
            x = base_model(x)
            pred_losses.append(model_loss(x))
        pred_losses = torch.stack(pred_losses).squeeze()
        pred_losses = pred_losses.mean(dim=1)   
        predicted_probs = F.softmax(-pred_losses, dim=0)

        loss = criterion(predicted_probs, relative_real_losses)
        epoch_loss += loss.item()
        cnt += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            epoch_loss_avg = epoch_loss / cnt
            print('Epoch:{}/{}; Epoch Loss = {:.5f}'
                  .format(int(iter_num // len(loader_train)), 
                          int(max_iter // len(loader_train)), 
                          epoch_loss_avg))

            PATH_TO_LOSSPRE_MODEL = '/PATH/TO/SAVE/MODEL/'
            CHECKPOINT_DIR = PATH_TO_LOSSPRE_MODEL + str(args.SEED) + "/EEGNet/New_loss_model_augment/"
            path = CHECKPOINT_DIR + args.data + "/s" + str(args.testID)
            os.makedirs(path, exist_ok=True)
            torch.save(model_loss.state_dict(), path + "/LossPredictor_epoch_" + str(iter_num) + ".pth")
        

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6, 7'
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
        if data_name == 'Seed':
            args.lr = 0.001

        # train batch size
        args.batch_size = 32
        
        if data_name == 'Driving':
            args.max_epoch = 20
        if data_name == 'Seed':
            args.max_epoch = 20

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

        for testID in range(N):
            # target subject
            args.testID = testID
            src_data, src_label = get_trainset(EEG, LABEL, args)
            print("Source Data Shape: ", src_data.shape, "Source Label Shape: ", src_label.shape)

            SEED = 44
            args.SEED = SEED
            fix_random_seed(SEED)

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
            base_dir = PATH_TO_MODEL + str(args.SEED) + '/EEGNet/New_EEGNetBlock/'
            tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/EEGNetBlock_epoch_100.pth'
            checkpoint = torch.load(tar_model_dir)
            base_model.load_state_dict(checkpoint)
            base_model = base_model.cuda()

            base_dir = base_dir = PATH_TO_MODEL + str(args.SEED) + '/EEGNet/New_Regression_head/'
            tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/Regression_head_epoch_100.pth'
            checkpoint = torch.load(tar_model_dir)
            regression_model.load_state_dict(checkpoint)
            regression_model = regression_model.cuda()

            F2 = 16
            input_dim = F2 * (eeg_length // (4 * 8))
            model_loss = LossPredictor(input_dim)
            model_loss = model_loss.cuda()

            learn_dropout_loss(model_loss, base_model, regression_model, src_data, src_label, args)
            # learn_augment_loss(model_loss, base_model, regression_model, src_data, src_label, args)