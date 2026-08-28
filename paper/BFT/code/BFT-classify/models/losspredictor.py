# The ranking module r of Section III-C and its training, for classification.
# Equation numbers refer to the paper "Backpropagation-Free Test-Time Adaptation
# for Lightweight EEG-Based BCIs" (IEEE J. Biomed. Health Inform., 2026):
#
#   Eq. (5)  w_{i,k} = softmax_k( r(z_i^(k)) )        reliability weights
#   Eq. (6)  L_ranking = || m(w_i) - pi_i ||_1        objective for r
#
# The mapping module m lives in sodeep/ and is loaded here through load_sorter.
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
# this module sits in models/, and imports both its sibling EEGNet and the
# augment module one level up.  Putting both directories on sys.path here means
# the file works when imported as models.losspredictor from BFT-classify/, not
# only when models/ happens to be the working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(os.path.dirname(_HERE))
from EEGNet import EEGNet, EEGNet_Block, EEGNet_Classifier, fix_random_seed
from augment import *

import numpy as np

# resolved from this file rather than from the working directory, so that
# sodeep/ is found no matter where the training script is launched from
PATH_TO_SODDEP = os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'sodeep')
sys.path.append(PATH_TO_SODDEP)
from sodeep import load_sorter, SpearmanLoss

    
# The ranking module r: a three-layer fully-connected network that maps a
# feature vector z (the flattened EEGNet block output) to one scalar.  It is
# trained to predict the task loss of that branch, so a SMALL output means a
# RELIABLE branch; every call site therefore takes softmax(-r(z)) for Eq. (5).
class EEGNetLossPredictor(nn.Module):
    def __init__(self, F2, Samples):
        super().__init__()
        self.F2 = F2
        self.Samples = Samples
        self.losspre_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                    out_features=(self.F2 * (self.Samples // (4 * 8))) // 2,
                    bias=True),
            nn.ELU(),
            nn.Linear(in_features=(self.F2 * (self.Samples // (4 * 8))) // 2,
                    out_features=(self.F2 * (self.Samples // (4 * 8))) // 4,
                    bias=True),  
            nn.ELU(),      
            nn.Linear(in_features=(self.F2 * (self.Samples // (4 * 8))) // 4,
                    out_features=1,
                    bias=True))

    def forward(self, x):
        output = self.losspre_block(x)
        return output


# Task-based rank labels pi_i of Eq. (6), for BFT-A.  NOTE: nn.CrossEntropyLoss
# reduces over the batch, so this returns ONE loss per branch for the whole
# mini-batch, not one per sample; the ranking supervision is therefore at
# batch level rather than the instance level of Eq. (6).
def compute_real_losses(augmented_inputs, model_target):
    loss_fn = nn.CrossEntropyLoss()
    real_losses = []
    model_target.eval()
    with torch.no_grad():
        for x_aug, label in augmented_inputs:
            pred = model_target(x_aug)
            loss = loss_fn(pred, label)
            real_losses.append(loss.item())
    return torch.tensor(real_losses)  # dim = 12


PATH_TO_LOSSPRE_MODEL = "/PATH/TO/SAVE/MODEL/"
PATH_TO_LOSSPRE_MODEL_DROPOUT = "/PATH/TO/SAVE/MODEL/"
# Train the ranking module r for BFT-A, the "Ranking Module Training" block of
# Algorithm 1.  For each mini-batch: build the K branches of Eq. (1), measure
# their true task losses, and fit r so that its Softmax profile matches the
# Softmax profile of the negated true losses.
def learn_augment_loss(model_loss, model_target, block_model, X_train, labels_train, args):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.long)

    data_train = torch.utils.data.TensorDataset(X_train, labels_train)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, 
                                               shuffle=True, drop_last=True)
    
    # Eq. (6).  NOTE: SpearmanLoss defaults to lbd = 0, and its forward is
    # lbd * MSE(m(w), rank(pi)) + (1 - lbd) * L1(w, pi).  At lbd = 0 the mapping
    # module m is evaluated but multiplied by zero, so the objective actually
    # optimised is a plain L1 between the two Softmax profiles and the ranking
    # space of Eq. (6) is never used.  Pass lbd=1 to put m back in the loop.
    sorter_checkpoint_path = PATH_TO_SODDEP + '/weights/12th_100epochs_best_model.pth.tar'
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

    model_target.eval()
    block_model.eval()
    model_loss.train()
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

        x_aug_list = generate_augmented_inputs(inputs_train, labels_train, args)

        # target profile pi_i: Softmax over the negated true task losses, so a
        # low-loss branch gets a high target weight
        real_losses = compute_real_losses(x_aug_list, model_target)
        relative_real_losses = F.softmax(-real_losses, dim=0)
        relative_real_losses = relative_real_losses.cuda()

        # predicted profile w_i, Eq. (5), with the same negation.  The
        # .mean(dim=1) averages r over the mini-batch, which is what makes the
        # supervision batch-level rather than per-sample.
        pred_losses = []
        for i in range(len(x_aug_list)):
            x, _ = x_aug_list[i]
            x = block_model(x)
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
            print('Epoch:{}/{}; Epoch Loss = {:.2f}'
                  .format(int(iter_num // len(loader_train)), 
                          int(max_iter // len(loader_train)), 
                          epoch_loss_avg))

            CHECKPOINT_DIR = PATH_TO_LOSSPRE_MODEL + str(args.SEED) + "/loss_model_new(batch=16)/"
            path = CHECKPOINT_DIR + args.data + "/s" + str(args.idt) + "/loss_pre"
            if os.path.isdir(path):
                 pass
            else:
                 os.makedirs(path)
            # name the checkpoint by EPOCH, not by iteration, so it matches the
            # EEGNetLossPredictor_epoch_20.pth that test.py loads
            torch.save(model_loss.state_dict(), 
                       path + "/EEGNetLossPredictor_epoch_" + str(int(iter_num // len(loader_train))) + ".pth")
            

# Train the ranking module r for BFT-D, the "Ranking Module Training" block of
# Algorithm 1 with the mask bank of Eq. (2) in place of the augmentations.
def learn_dropout_loss(model_loss, block_model, classifier, X_train, labels_train, args):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.long)

    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_train = X_train[:, :, :eeg_length]
    X_train = X_train.unsqueeze(1)
    X_train, labels_train = X_train.cuda(), labels_train.cuda()

    data_train = torch.utils.data.TensorDataset(X_train, labels_train)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, 
                                               shuffle=True, drop_last=True)
    
    # Eq. (6), same lbd = 0 caveat as in learn_augment_loss above.
    # NOTE: the path separator is missing, so this resolves to
    # '../sodeepweights/...' and the checkpoint cannot be found as written.
    sorter_checkpoint_path = PATH_TO_SODDEP + '/weights/10th_100epochs_best_model.pth.tar'
    loss_fn = nn.CrossEntropyLoss()
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

    block_model.eval()
    classifier.eval()
    model_loss.train()

    # The K masks of Eq. (2), identical to the bank used at test time
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
        output1 = block_model(inputs_train)
        B, D = output1.shape
        for (start_r, end_r), key in zip(drop_ranges, range_keys):
            output1_mask = output1.clone()
            start = int(start_r * D)
            end = int(end_r * D)
            output1_mask[:, start:end] = 0.0

            # B * 2
            this_outputs = classifier(output1_mask)
            this_loss = loss_fn(this_outputs, labels_train)
            dropout_loss_list.append(this_loss.item())

            pred_losses.append(model_loss(output1_mask))
        # target profile pi_i and predicted profile w_i of Eq. (5), both length
        # K.  As in learn_augment_loss, this_loss is a batch-mean cross entropy
        # and pred_losses is averaged over the batch, so the ranking is
        # supervised per mini-batch rather than per sample.
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
            print('Epoch:{}/{}; Epoch Loss = {:.2f}'
                  .format(int(iter_num // len(loader_train)), 
                          int(max_iter // len(loader_train)), 
                          epoch_loss_avg))

            CHECKPOINT_DIR = PATH_TO_LOSSPRE_MODEL_DROPOUT + str(args.SEED) + "/loss_model_dropout_new(batch=16)/"
            path = CHECKPOINT_DIR + args.data + "/s" + str(args.idt) + "/loss_pre"
            if os.path.isdir(path):
                 pass
            else:
                 os.makedirs(path)
            # name the checkpoint by EPOCH, not by iteration, so it matches the
            # EEGNetLossPredictor_epoch_20.pth that test.py loads
            torch.save(model_loss.state_dict(), 
                       path + "/EEGNetLossPredictor_epoch_" + str(int(iter_num // len(loader_train))) + ".pth")


if __name__ == '__main__':
    sorter_checkpoint_path = PATH_TO_SODDEP + '/weights/12th_50epochs_best_model.pth.tar'
    criterion = SpearmanLoss(*load_sorter(sorter_checkpoint_path))
    a = [2, 3, 3.2, 1.5, 3.4, 6.7, 9, 1.1, 12.1, 11, 57, 100]
    b = [6.7, 9, 1.1, 12.1, 11, 57, 2, 3, 3.2, 1.5, 3.4, 100]
    a = torch.tensor(a)
    b = torch.tensor(b)
    print(a.shape)
    loss = criterion(a, b)
    print(loss)
    