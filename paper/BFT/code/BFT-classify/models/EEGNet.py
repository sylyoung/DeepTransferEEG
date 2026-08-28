import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import accuracy_score

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from augment_trainset import data_aug_random


def fix_random_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


# EEGNet-v4
class EEGNet(nn.Module):
    # default values:
    # kernLength = sample_rate / 2
    # F1 = 8, D = 2, F2 = 16
    # dropoutRate = 0.5 for within-subject, = 0.25 for cross-subject
    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLength: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate:  float):
        super(EEGNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLength // 2 - 1,
                          self.kernLength - self.kernLength // 2, 0,
                          0)),
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLength),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                    out_features=self.n_classes,
                    bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        return output
    

class EEGNet_Block(nn.Module):
    def __init__(self, block1, block2):
        super().__init__()
        self.block1 = block1
        self.block2 = block2

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        return x


class EEGNet_Classifier(nn.Module):
    def __init__(self, classifier_block):
        super().__init__()
        self.classifier_block = classifier_block

    def forward(self, x):
        return self.classifier_block(x)
    

PATH_EEGNET_MODEL = "/PATH/TO/SAVE/MODEL/"
PATH_EEGNET_MODEL_NOAUG = "/PATH/TO/SAVE/MODEL/"

def train_EEGNet(base_network, X_train, labels_train, X_test, labels_test, args):
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_test = X_test[:, :, :eeg_length]
    X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)
    X_test = X_test.unsqueeze(1)
    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, 
                                              shuffle=False, drop_last=False)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.long)
    data_train = torch.utils.data.TensorDataset(X_train, labels_train)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, 
                                               shuffle=True, drop_last=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_network.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(loader_train)
    # max(1, ...): the unguarded expression is 0 for any max_epoch below 10,
    # and iter_num % 0 raises ZeroDivisionError on the first iteration
    interval_iter = max(1, int(args.max_epoch / 10) * max_iter // args.max_epoch)
    iter_num = 0
    epoch_loss = 0
    cnt = 0

    start_train = True
    base_network.train()
    while iter_num < max_iter:
        try:
            inputs_train, labels_train = next(iter_train)
        except:
            iter_train = iter(loader_train)
            inputs_train, labels_train = next(iter_train)
            last_inputs_train, last_labels_train = inputs_train, labels_train

        if inputs_train.size(0) == 1:
            continue

        iter_num += 1

        inputs_train = inputs_train.detach().cpu().numpy()
        labels_train = labels_train.detach().cpu().numpy()

        last_inputs_train_c, last_labels_train_c = last_inputs_train, last_labels_train
        last_inputs_train, last_labels_train = inputs_train, labels_train

        inputs_train, labels_train = data_aug_random(inputs_train, last_inputs_train_c, labels_train, last_labels_train_c, args)

        inputs_train = inputs_train[:, :, :eeg_length]
        inputs_train = torch.tensor(inputs_train, dtype=torch.float32)
        labels_train = torch.tensor(labels_train, dtype=torch.long)
        inputs_train = inputs_train.unsqueeze(1)
        inputs_train, labels_train = inputs_train.cuda(), labels_train.cuda()

        outputs_train = base_network(inputs_train)

        if start_train:
            all_train_output = outputs_train.float().cpu()
            all_train_label = labels_train.float()
            start_train = False
        else:
            all_train_output = torch.cat((all_train_output, outputs_train.float().cpu()), 0)
            all_train_label = torch.cat((all_train_label, labels_train.float()), 0)

        classifier_loss = criterion(outputs_train, labels_train)

        epoch_loss += classifier_loss.item()
        cnt += 1

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            with torch.no_grad():
                iter_test = iter(loader_test)
                for i in range(len(loader_test)):
                    data = next(iter_test)
                    inputs = data[0]
                    labels = data[1]
                    inputs = inputs.cuda()

                    outputs = base_network(inputs)
                    if i == 0:
                        all_output = outputs.float().cpu()
                        all_label = labels.float()
                    else:
                        all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                        all_label = torch.cat((all_label, labels.float()), 0)
            all_train_output = nn.Softmax(dim=1)(all_train_output)
            _, train_predict = torch.max(all_train_output, 1)
            train_pred = torch.squeeze(train_predict).float()
            train_true = all_train_label.cpu()
            train_acc = accuracy_score(train_true, train_pred) * 100

            all_output = nn.Softmax(dim=1)(all_output)
            _, predict = torch.max(all_output, 1)
            pred = torch.squeeze(predict).float()
            true = all_label.cpu()
            acc = accuracy_score(true, pred) * 100

            epoch_loss_avg = epoch_loss / cnt
            start_train = True
            print('Epoch:{}/{}; Test Acc = {:.2f}; Train Acc = {:.2f}; Epoch Loss = {:.2f}'
                  .format(int(iter_num // len(loader_train)), 
                          int(max_iter // len(loader_train)), 
                          acc, train_acc, epoch_loss_avg))

            CHECKPOINT_DIR = PATH_EEGNET_MODEL + str(args.SEED) + "/EEGNet_pth/"
            path = CHECKPOINT_DIR + args.data + "/s" + str(args.idt)
            os.makedirs(path, exist_ok=True)
            # name the checkpoint by EPOCH, not by iteration.  iter_num counts
            # mini-batches, so this wrote EEGNet_epoch_7200.pth while test.py and
            # quantization.py both load EEGNet_epoch_200.pth, a file the trainer
            # never produced.
            torch.save(base_network.state_dict(), 
                       path + "/EEGNet_epoch_" + str(int(iter_num // len(loader_train))) + ".pth")
            base_network.train()
            epoch_loss = 0
            cnt = 0


def train_EEGNet_noaug(base_network, X_train, labels_train, X_test, labels_test, args):
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_test = X_test[:, :, :eeg_length]
    X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)
    X_test = X_test.unsqueeze(1)
    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, 
                                              shuffle=False, drop_last=False)

    X_train = X_train[:, :, :eeg_length]
    X_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.long)
    X_train = X_train.unsqueeze(1)
    data_train = torch.utils.data.TensorDataset(X_train, labels_train)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, 
                                               shuffle=True, drop_last=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_network.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(loader_train)
    # max(1, ...): the unguarded expression is 0 for any max_epoch below 10,
    # and iter_num % 0 raises ZeroDivisionError on the first iteration
    interval_iter = max(1, int(args.max_epoch / 10) * max_iter // args.max_epoch)
    iter_num = 0
    epoch_loss = 0
    cnt = 0

    start_train = True
    base_network.train()
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
        outputs_train = base_network(inputs_train)

        if start_train:
            all_train_output = outputs_train.float().cpu()
            all_train_label = labels_train.float()
            start_train = False
        else:
            all_train_output = torch.cat((all_train_output, outputs_train.float().cpu()), 0)
            all_train_label = torch.cat((all_train_label, labels_train.float()), 0)

        classifier_loss = criterion(outputs_train, labels_train)

        epoch_loss += classifier_loss.item()
        cnt += 1

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            with torch.no_grad():
                iter_test = iter(loader_test)
                for i in range(len(loader_test)):
                    data = next(iter_test)
                    inputs = data[0]
                    labels = data[1]
                    inputs = inputs.cuda()

                    outputs = base_network(inputs)
                    if i == 0:
                        all_output = outputs.float().cpu()
                        all_label = labels.float()
                    else:
                        all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                        all_label = torch.cat((all_label, labels.float()), 0)
            all_train_output = nn.Softmax(dim=1)(all_train_output)
            _, train_predict = torch.max(all_train_output, 1)
            train_pred = torch.squeeze(train_predict).float()
            train_true = all_train_label.cpu()
            train_acc = accuracy_score(train_true, train_pred) * 100

            all_output = nn.Softmax(dim=1)(all_output)
            _, predict = torch.max(all_output, 1)
            pred = torch.squeeze(predict).float()
            true = all_label.cpu()
            acc = accuracy_score(true, pred) * 100

            epoch_loss_avg = epoch_loss / cnt
            start_train = True
            print('Epoch:{}/{}; Test Acc = {:.2f}; Train Acc = {:.2f}; Epoch Loss = {:.2f}'
                  .format(int(iter_num // len(loader_train)), 
                          int(max_iter // len(loader_train)), 
                          acc, train_acc, epoch_loss_avg))

            CHECKPOINT_DIR = PATH_EEGNET_MODEL_NOAUG + str(args.SEED) + "/EEGNet_pth/"
            path = CHECKPOINT_DIR + args.data + "/s" + str(args.idt)
            os.makedirs(path, exist_ok=True)
            # name the checkpoint by EPOCH, not by iteration.  iter_num counts
            # mini-batches, so this wrote EEGNet_epoch_7200.pth while test.py and
            # quantization.py both load EEGNet_epoch_200.pth, a file the trainer
            # never produced.
            torch.save(base_network.state_dict(), 
                       path + "/EEGNet_epoch_" + str(int(iter_num // len(loader_train))) + ".pth")
            base_network.train()
            epoch_loss = 0
            cnt = 0