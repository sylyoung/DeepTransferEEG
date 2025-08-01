# -*- coding: utf-8 -*-
# @Time    : 2025/08/01
# @Author  : Siyang Li
# @File    : EEGNet_demo.py
# This one-file-code serves as a simple demo for those who are not particularly familiar with EEG decoding, Python, deep learning
# Comments will help you understand each line of code
# This file considers EEG data from the Motor Imagery paradigm, to decode the movement intentions of subjects/users, using deep learning
# This file randomly created the EEG data, if the data are not given at path ./args.data_name/X.npy numpy array of shape (num_trials, num_channels, num_timesamples) such as (1000, 59, 1000). Similarly for labels at path /args.data_name/labels.npy of shape (num_trials, )
# You could download the examplar EEG data at https://www.bbci.de/competition/iv/ , after application

# We apply a pipeline of
# 1) specify hyperparameters of the experiments
# 2) EEG loading (if not supplied, create random EEG data)
# 3) go into each individual experiment (multiple experiments must be run for each subject and for each random seed)
# 4) apply Euclidean Alignment (if needed)
# 5) apply Channel Reflection data augmentation (if needed)
# 6) initialize and train the network, output training loss and test set accuracy during each epoch
# 7) output final performance (test set accuracy after last epoch)


######################## this part imports all the libraries required to run this code #################################
# install these dependecies using 'pip install xxx' (recommended), or via a conda virtual environment
# note that sklearn is installed via 'pip install scikit-learn', not sklearn
from sklearn.metrics import accuracy_score
from scipy.linalg import fractional_matrix_power
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# these do not need to be installed, they come with Python
import argparse
import os
import gc
import sys
import random
from pathlib import Path
########################################################################################################################
# after this import part is done, the main function is executed, at line 608


# EEGNet-v4
# This is a lightweight (with around 1,000-5,000 trainable parameters, depending on input size) convolutional neural network architecture, fit for small data EEG decoding
# It is an end-to-end decoding architecture, and takes the trial-wise !RAW! EEG signals as input, NOT extracted EEG features
# It is more suited to Motor Imagery and P300 decoding, and is applicable to also Seizure Detection, than to other paradigms, as far as I know
# The input shape to this network is (number_samples_in_a_mini_batch, 1, number_electrodes, number_time_samples), the 1 is there to keep with the form of pytorch CNN implementations
# reference: "EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces", JNE, 2018
class EEGNet(nn.Module):
    # When you first call EEGNet(), this initialization function is FIRST executed, to create a new EEGNet class object
    # n_classes: number of classes for classification
    # Chans: number of electrodes
    # Samples: number of time samples (NOT number of samples in machine learning term!)
    # kernLength: the length of CNN kernel in temporal domain, should be set to sampling_rate, i.e., 500Hz, divided by 2 for half a second temporal kernel, kernLength = 250
    # F1, D, F2, the hyperparameters for CNN kernels, recommended to set as default: F1 = 8, D = 2, F2 = 16
    # dropoutRate, recommended to set as default, = 0.5 for within-subject, = 0.25 for cross-subject decoding, as suggested in original paper
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

        # These are two CNN blocks for feature extraction
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

        # This is a fully-connected linear layer of classification block
        # it outputs self.n_classes neurons of continuous values
        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                    out_features=self.n_classes,
                    bias=True))

    # This function is not called when you first create EEGNet() class object
    # It is called when you forward EEG data into the network, i.e., EEGNet(4_d_eeg_data), where 4_d_eeg_data.shape = (number_samples_in_a_mini_batch, 1, number_electrodes, number_time_samples)
    # And pytorch automatically handles the backward backpropagation process for updating the parameters of the network
    # backpropagation is achieved with a few lines of code, detailing down below
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)
        # print(output.shape) # you could print the output.shape here, to see what the deep feature dimensionality is
        output = self.classifier_block(output)
        return output  # the final output is of shape (number_samples_in_a_mini_batch, self.n_classes), but the values are continuous in range (-infinity, infinity), not the prediction PROBABILITIES

# this determines the random seeds (for random mini-batch sampling, random model weight initialization, etc.)
# multiple runs of experiments of different seeds should be run, so that different performance scores can be get, so that variance of the algorithm can be calculated
def fix_random_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


# Euclidean Alignment
# "Transfer learning for brain–computer interfaces: A Euclidean space data alignment approach", IEEE TBME, 2020
# This approach normalizes each subject/user's data, so that each subject's mean covariance matrix will be an identity matrix
# Thus model trained using data of other subjects can perform better when classifying target subject/user's EEG trial
def EA(x, epsilon=1e-6):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    try:
        sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    except:
        to_add = np.eye(len(refEA)) * epsilon
        sqrtRefEA = fractional_matrix_power(refEA + to_add, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


# Euclidean Alignment
# arithmetic mean only, SPD-safe
def EA_SPDsafe(x, epsilon=1e-6):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    n = len(x)
    C = np.zeros((x[0].shape[0], x[0].shape[0]))
    for X in x:
        C += X @ X.T
    R_bar = C / n
    trace = np.trace(R_bar)
    R_bar += epsilon * (trace / R_bar.shape[0]) * np.eye(R_bar.shape[0])

    eigvals, eigvecs = np.linalg.eigh(R_bar)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    ref = eigvecs @ D_inv_sqrt @ eigvecs.T

    XEA = ref @ x

    return XEA


# Euclidean Alignment for online(incremental)
# arithmetic mean only, SPD-safe
# and online update for target subject, fit for real-time applications
# "T-TIME: Test-Time Information Maximization Ensemble for Plug-and-Play BCIs", IEEE TBME, 2024
def EA_SPDsafe_online(x, R_bar=None, num_samples=0, epsilon=1e-6):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    R_bar : numpy array
        data of shape (num_channels, num_channels)
    num_samples : int

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    n = len(x)

    n += num_samples

    if R_bar is None:
        R_bar = 0

    C = np.zeros((x[0].shape[0], x[0].shape[0]))
    for X in x:
        C += X @ X.T
    R_bar += C / n
    trace = np.trace(R_bar)
    R_bar += epsilon * (trace / R_bar.shape[0]) * np.eye(R_bar.shape[0])

    eigvals, eigvecs = np.linalg.eigh(R_bar)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    ref = eigvecs @ D_inv_sqrt @ eigvecs.T

    XEA = ref @ x

    return XEA

# This function executes EA offline
def data_alignment_offline(X, num_subjects, trials_arr=None, target_id=None):
    '''
    :param X: np array, EEG data (num_trials, num_channels, num_timesamples)
    :param num_subjects: int, number of total subjects in X
    :param trials_arr: list, use if each subject has unequal number of trials, e.g., [200, 300, 200, 200]
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    if trials_arr is None:
        out = []
        for i in range(num_subjects):
            subj_x = EA_SPDsafe(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
            out.append(subj_x)
        XEA = np.concatenate(out, axis=0)
    else:
        out = []
        past_target = 0
        for i in range(len(trials_arr) - 1):
            if target_id == i:
                past_target = 1
                continue
            start = int(np.sum(trials_arr[:i - past_target]))
            end = int(np.sum(trials_arr[:i + 1 - past_target]))
            print('start', start, 'end', end)
            subj_x = EA_SPDsafe(X[start:end, :, :])
            print('subj', i+1, subj_x.shape)
            out.append(subj_x)
        XEA = np.concatenate(out, axis=0)
        assert len(XEA) == len(X), print('wrong trials_arr!')
    return XEA

# This function executes EA online
def data_alignment_online(X):
    '''
    :param X: np array, EEG data (num_trials, num_channels, num_timesamples)
    :return: np array, aligned EEG data
    '''

    # Online(Incremental) EA
    # Much proper way to do EA for target subject considering online BCIs
    Xt_aligned = []
    R = 0
    num_samples = 0
    for ind in range(len(X)):
        curr = X[ind]
        cov = np.cov(curr)
        # Note that the following line is an update of the mean covariance matrix (R), instead of a full recalculation. It is much faster computation in this way.
        # Note also that the covariance matrix calculation should take in all visible samples(trials) for this domain(subject)
        R = (R * num_samples + cov) / (num_samples + 1)
        num_samples += 1
        sqrtRefEA = fractional_matrix_power(R, -0.5)
        # transform the original trial. All latter algorithms only use the transformed data as input
        curr_aligned = np.dot(sqrtRefEA, curr)
        Xt_aligned.append(curr_aligned)
    Xt_aligned = np.array(Xt_aligned)
    # EA done

    return Xt_aligned


def traintest_split_cross_subject(X, y, num_subjects, test_subject_id, trials_arr=None):
    # assumes each subject has unequal number of trials
    # trials_arr is a numpy array with number denoting number of trials for each subject
    if trials_arr is not None:
        accum_arr = []
        for t in range(len(trials_arr)):
            accum_arr.append(np.sum([trials_arr[:(t + 1)]]))
        print(accum_arr)
        data_subjects = np.split(X, indices_or_sections=accum_arr, axis=0)
        labels_subjects = np.split(y, indices_or_sections=accum_arr, axis=0)
    # if each subject has equal number of trials
    else:
        data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
        labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    test_x = data_subjects.pop(test_subject_id)
    test_y = labels_subjects.pop(test_subject_id)

    train_x = np.concatenate(data_subjects, axis=0)
    train_y = np.concatenate(labels_subjects, axis=0)
    print('Test subject s' + str(test_subject_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


# Channel Reflection
# "Channel reflection: Knowledge-driven data augmentation for EEG-based brain–computer interfaces"
# This function applies data augmentation (2 x training data) for left/right hand EEG motor imagery classification
def CR_transform(X, left_mat, right_mat):
    """

    Parameters
    ----------
    X: torch tensor of shape (num_samples, 1, num_channels, num_timesamples)
    left_mat: numpy array of shape (a, ), where a is the number of left brain channels, in order
    right_mat: numpy array of shape (b, ), where b is the number of right brain channels, in order

    Returns
    -------
    transformedX: transformed signal of torch tensor of shape (num_samples, num_channels, num_timesamples)
    """

    num_samples, _, num_channels, num_timesamples = X.shape
    transformedX = torch.zeros((num_samples, 1, num_channels, num_timesamples))
    for ch in range(num_channels):
        if ch in left_mat:
            ind = left_mat.index(ch)
            transformedX[:, 0, ch, :] = X[:, 0, right_mat[ind], :]
        elif ch in right_mat:
            ind = right_mat.index(ch)
            transformedX[:, 0, ch, :] = X[:, 0, left_mat[ind], :]
        else:
            transformedX[:, 0, ch, :] = X[:, 0, ch, :]

    return transformedX


# This is the main function for training an EEG motor imagery classification decoding model
def train_nn(args):

    ####################################Load Data from File#######################################
    try:
        X = np.load('./' + args.data_name + '_X.npy')  # numpy array of (num_trials, num_channels, num_timesamples)  (1000, 59, 1000)
    except:
        # print('please put signal under path: ', './' + args.data_name + '_X.npy')
        print('IMPORTANT! did not find signal data at path: ', './' + args.data_name + '_X.npy')
        print('IMPORTANT! creating random signal trials...')
        X = np.random.randn(1000, 59, 1000)
    try:
        y = np.load('./' + args.data_name + '_labels.npy')
    except:
        # print('please put labels under path: ', './' + args.data_name + '_labels.npy')
        print('IMPORTANT! did not find labels at path: ', './' + args.data_name + '_labels.npy')
        print('IMPORTANT! creating random labels...')
        y = np.random.randint(0, 2, size=(1000,))

    assert len(X.shape) == 3, print('signal must be in shape (num_trials, num_channels, num_timesamples)')
    assert len(y.shape) == 1, print('labels must be in shape (num_trials, )')  # note that labels shape is (num_trials, ), reshape your labels into (-1, ) to build this shape

    ####################################Build Training/Test Sets#######################################
    # cross refers to cross-subject decoding, where training data are from other subjects
    if args.setting == 'cross':
        # assumes each subject has equal number of trials
        X_src, y_src, X_tar, y_tar = traintest_split_cross_subject(X, y, args.N, args.idt, trials_arr=None)
        # EA could be coded before subject-wise transfer task, to save computational cost in repeated experiments
        print('executing cross-subject decoding')
        if args.align:
            print('using EA')
            # trial_array is the number of trials for each subject
            X_src = data_alignment_offline(X_src, args.N - 1, trials_arr=[200,200,200,200,200])
            X_tar = EA_SPDsafe_online(X_tar)
    # within refers to within-subject decoding, where training data are from the same subject
    elif args.setting == 'within':
        X_src, y_src, X_tar, y_tar = traintest_split_cross_subject(X, y, args.N, args.idt, trials_arr=None)
        # an example: within-subject task splits target subject into two equal splits for training and test (50% and 50%)
        print('spliting target subject half-half for training/test')
        X_src, y_src, X_tar, y_tar = X_tar[:int(len(X_tar) // 2)], y_tar[:int(len(y_tar) // 2)], X_tar[int(len(X_tar) // 2):], y_tar[int(len(y_tar) // 2):]
        if args.align:
            print('using EA')
            # SPD-safe implementation
            X_src = data_alignment_offline(X_src, 1)
            X_tar = EA_SPDsafe_online(X_tar)

    # src, tar refers to source (training) and target (test) subject
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)

    # Channel Reflection data augmentation, only applies to left/right hand EEG motor imagery classification
    if args.class_num == 2:
        if args.CR_aug:
            if args.data_name == 'MI1':
                left_mat = [0, 2, 3, 4, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 33, 34, 35, 36, 41, 42, 43, 48, 49,
                            50, 55, 57]
                right_mat = [1, 8, 7, 6, 15, 14, 13, 23, 22, 21, 20, 32, 31, 30, 29, 40, 39, 38, 37, 47, 46, 45, 54, 53,
                             52, 56, 58]
                aug_train_x = CR_transform(
                    torch.from_numpy(X_src).to(torch.float32).reshape(X_src.shape[0], 1, args.chn, -1),
                    left_mat, right_mat).numpy().reshape(X_src.shape[0], args.chn, -1)
                aug_train_y = 1 - y_src

                X_src = np.concatenate((X_src, aug_train_x))
                y_src = np.concatenate((y_src, aug_train_y))
                print('after CR augmentation X_src.shape, y_src.shape:', X_src.shape, y_src.shape)

    Xs, Ys = torch.from_numpy(X_src).to(
        torch.float32), torch.from_numpy(y_src.reshape(-1, )).to(torch.long)
    if args.backbone == 'EEGNet':
        # convert data from shape (num_trials/batch_size, num_channels, num_timesamples)
        # to shape                (num_trials/batch_size, 1, num_channels, num_timesamples)
        # since EEGNet takes 4-D input
        Xs = Xs.unsqueeze_(1)

    # convert numpy data to pytorch data for deep learning
    Xt, Yt = torch.from_numpy(X_tar).to(
        torch.float32), torch.from_numpy(y_tar.reshape(-1, )).to(torch.long)
    if args.backbone == 'EEGNet':
        Xt = Xt.unsqueeze_(1)

    # put data from CPUs to GPUS
    if args.data_env != 'local':
        Xs, Ys, Xt, Yt = Xs.cuda(), Ys.cuda(), Xt.cuda(), Yt.cuda()

    # create pytorch tensor dataloaders
    data_src = torch.utils.data.TensorDataset(Xs, Ys)
    data_tar = torch.utils.data.TensorDataset(Xt, Yt)
    dset_loaders = {}
    # source training dataset shuffles training data, drops last mini-batch
    dset_loaders["source"] = torch.utils.data.DataLoader(data_src, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # target test dataset keeps original trial order, keeps all trials
    dset_loaders["target"] = torch.utils.data.DataLoader(data_tar, batch_size=args.batch_size, shuffle=False, drop_last=False)

    ####################################Model Setting#######################################
    # this call initializes EEGNet model (call the __init__ function of EEGNet)
    base_network = EEGNet(n_classes=args.class_num,
                        Chans=args.chn,
                        Samples=args.time_sample_num,
                        kernLength=int(args.sample_rate // 2),
                        F1=F1,
                        D=D,
                        F2=F2,
                        dropoutRate=0.25 if args.setting == 'cross' else 0.5)  # dropout is 0.25 for cross-subject and 0.5 for within-subject, as suggested by EEGNet paper.
    # also needs to put neural network weights to GPU, if GPU possible
    # data and model weights must be on same device for matrix opearation
    if args.data_env != 'local':
        base_network = base_network.cuda()

    ####################################Optimization#######################################
    print('start training...')

    # define learning/optimization objective
    # Empirical Risk Minimization
    # in the form of Cross-Entropy Loss
    # suitable for classification
    # for class-imbalanced training, add class-based weights
    criterion = nn.CrossEntropyLoss()

    # optimizer, which defines how mini-batch gradient-descent backpropagation is done
    # args.lr is the learning rate
    optimizer = optim.Adam(base_network.parameters(), lr=args.lr)

    # epochs of learning
    # each epoch is number of batches in the training set dataloader
    # e.g., epoch = 50, batch_size = 32, training_set_size = 1000. Total update step is (1000 // 32) * 50
    max_iter = args.epoch * len(dset_loaders["source"])
    interval_iter = int(args.epoch / 10) * max_iter // args.epoch
    args.max_iter = max_iter
    iter_num = 0

    # this .train() call makes the weights backpropagation-able
    # set this during training phase
    base_network.train()

    epoch_loss = 0
    cnt = 0
    while iter_num < max_iter:

        # get a mini-batch of data and corresponding labels
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1

        # this is the forward call
        # the data are forward through the network
        # if network.train(), gradients will be calculated
        outputs_source = base_network(inputs_source)

        # calculate the loss value of this mini-batch
        # note that nn.CrossEntropyLoss() integrates outputs -> probabilities -> loss value
        # so no need to calculate probabilities, if you need probabilities output of the network, apply a softmax function of the outputs
        classifier_loss = criterion(outputs_source, labels_source)

        epoch_loss += classifier_loss.item()
        cnt += 1

        # this is the backpropagation update part, which applies a step of update of network weights
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        # we calculate the test set loss for each epoch of training
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            # set the network to evaluation setting
            # this de-activate Batch-Normalization and Dropout
            # MUST call this line for test
            base_network.eval()

            start_test = True
            test_loader = dset_loaders["target"]
            # no need to compute gradient functions, as we are testing
            with torch.no_grad():
                iter_test = iter(test_loader)
                for i in range(len(test_loader)):
                    data = next(iter_test)
                    inputs = data[0]
                    labels = data[1]
                    if args.data_env != 'local':
                        inputs = inputs.cuda()
                    outputs = base_network(inputs)
                    if start_test:
                        all_output = outputs.float().cpu()
                        all_label = labels.float()
                        start_test = False
                    else:
                        all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                        all_label = torch.cat((all_label, labels.float()), 0)

            # this is the softmax function which converts the continous outputs to probabilities
            all_output = nn.Softmax(dim=1)(all_output)

            # this is the class prediction
            _, predict = torch.max(all_output, 1)
            pred = torch.squeeze(predict).float()
            true = all_label.cpu()

            # convert the class prediction from pytorch tensor to numpy, so that sklearn function can apply for performance metric calculation
            acc = accuracy_score(true, pred) * 100
            epoch_loss_avg = epoch_loss / cnt

            # note that we calculate test set accuracy, and training set epoch-wise loss
            print('Epoch:{}/{}; Test Acc = {:.2f}; Epoch Loss = {:.2f}%'.format(int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc, epoch_loss_avg))

            # convert the network back to training mode for next epoch
            base_network.train()
            epoch_loss = 0
            cnt = 0

    print('Test Acc = {:.2f}'.format(acc))

    # use this section if you want to save the trained model weights to disk
    # you can load the model weights as well
    """
    ####################################save model parameters#######################################
    print('saving model checkpoint...')

    if not os.path.isdir('./runs/' + str(args.data_name) + '/'):
        path = Path('./runs/' + str(args.data_name) + '/')
        path.mkdir(parents=True)

    if args.align:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '.ckpt')
    else:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_noEA' + '.ckpt')
    """

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()

    return acc


# when you run EEGNet_demo.py, this is the function that would be called first, after imports
if __name__ == '__main__':

    # this are the hyperparameter configurations
    parser = argparse.ArgumentParser(description='NN experiment')
    parser.add_argument('--number_seeds', type=int, default=3, help='number of repeat experiments with different random seeds')
    parser.add_argument('--gpu_id', type=int, default=-1, help='GPU ID, -1 means CPU')
    parser.add_argument('--dataset_name', type=str, default='MI1', help='dataset name')
    parser.add_argument('--class_num', type=int, default=2, help='classification task: number of classes')
    parser.add_argument('--setting', type=str, default='cross', help='cross: cross-subject; within: within-subject by spliting target subject trials by half for 50:50 training/test')
    parser.add_argument('--epoch', type=int, default=30, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=50, help='mini batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--align', type=bool, default=True, help='use Euclidean Alignment or not')
    parser.add_argument('--CR_aug', type=bool, default=True, help='use Channel Reflection data augmentation or not')
    args = parser.parse_args()

    # 0-indexing
    dct = pd.DataFrame(columns=['dataset', 'avg', 'std', 'S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'])

    # N: number of subjects, chn: number of channels, time_sample_num: total number of time samples, sample_rate: sample rate in Hz
    # feature_deep_dim is feature dimensionality after CNN layers, this value vary across datasets and architecture. You could input the data through the CNN network to see what the dimension is. DO NOT recommend too big of a feature dimension (>1000)
    if args.dataset_name == 'MI1':  # https://www.bbci.de/competition/iv/  "The non-invasive Berlin Brain-Computer Interface: Fast acquisition of effective performance in untrained subjects." NeuroImage, 2007
        paradigm, N, chn, time_sample_num, sample_rate, feature_deep_dim = 'MI', 5, 59, 1000, 250, 248
    else:
        print('unknown dataset name')
        sys.exit(0)

    F1, D, F2 = 8, 2, 16

    args = argparse.Namespace(feature_deep_dim=feature_deep_dim, setting=args.setting, number_seeds=args.number_seeds,
                              time_sample_num=time_sample_num, sample_rate=sample_rate, align=args.align, gpu_id=args.gpu_id,
                              N=N, chn=chn, class_num=args.class_num, paradigm=paradigm, data_name=args.dataset_name,
                              F1=F1, D=D, F2=F2, epoch=args.epoch, lr=args.lr, batch_size=args.batch_size, CR_aug=args.CR_aug)

    args.method = 'EEGNet'
    args.backbone = 'EEGNet'

    print(args)

    # GPU device id
    try:
        device_id = str(args.gpu_id)
        print('device_id', device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        print('using GPU, device_id', device_id)
        args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
    except:
        args.data_env = 'local'
        print('no GPU found. using CPU')

    total_acc = []

    # multiple random seeds for repeated experiments
    for s in np.arange(args.number_seeds, dtype=int):
        args.SEED = s
        print('##############################random seed:', args.SEED, '##############################')
        fix_random_seed(args.SEED)
        torch.backends.cudnn.deterministic = True

        sub_acc_all = np.zeros(N)

        # treat each subject/user as target/test subject once
        for idt in range(N):
            args.idt = idt
            target_str = 'S' + str(idt)
            info_str = '=====================' + target_str + ' as Test Subject =========================='
            print(info_str)

            # this is main training function call
            acc = train_nn(args)
            sub_acc_all[idt] = acc

        # results are averaged across subjects/users
        print('Sub acc: ', np.round(sub_acc_all, 2))
        print('Avg acc: ', np.round(np.mean(sub_acc_all), 2))
        total_acc.append(sub_acc_all)

        acc_sub_str = str(np.round(sub_acc_all, 2).tolist())
        acc_mean_str = str(np.round(np.mean(sub_acc_all), 2).tolist())

        print()

    print('\n' + '#' * 20 + 'final results' + '#' * 20)

    print('total acc:', str(total_acc))

    # results are further averaged across repeated experiments
    subject_mean = np.round(np.average(total_acc, axis=0), 2)
    total_mean = np.round(np.average(np.average(total_acc)), 2)
    total_std = np.round(np.std(np.average(total_acc, axis=1)), 2)
    print('subject mean:', subject_mean)
    print('total mean:', total_mean)
    print('total std:', total_std)

    result_dct = {'dataset': args.data_name, 'avg': total_mean, 'std': total_std}
    for i in range(len(subject_mean)):
        result_dct['S' + str(i)] = subject_mean[i]

    dct = dct.append(result_dct, ignore_index=True)

    # save results to csv
    print('saving results to ./results.csv ...')
    dct.to_csv("./results.csv")