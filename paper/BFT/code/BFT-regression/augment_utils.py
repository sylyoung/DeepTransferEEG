# Knowledge-guided augmentations for BFT-A on the regression tasks.
# Equation numbers refer to the paper "Backpropagation-Free Test-Time Adaptation
# for Lightweight EEG-Based BCIs" (IEEE J. Biomed. Health Inform., 2026):
#
#   Eq. (1)  z_t^(k) = g(T_k(x_t))
#   Eq. (5)  w_{i,k} = softmax_k( r(z_i^(k)) )
#   Eq. (8)  y_t = mean of h(z_t^(k)) over the ceil(K/2) most reliable branches
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
from utils.regression_about import *


def test_BNadapt(base_model, regression_model, tar_data, tar_label, args, test_batch=64):
    base_model = base_model.cuda()
    regression_model = regression_model.cuda()
    base_model.eval()
    regression_model.eval()
    aug_dir = ["identity"]
    all_output = {}
    PATH_TO_AUGED_DATA = '/PATH/TO/AUGED/DATA/'
    data_auged_dir = PATH_TO_AUGED_DATA + args.data + '/s' + str(args.testID)
    for aug_way in aug_dir:
        X_test, labels_test = load_aug(data_auged_dir, aug_way)

        labels_test = labels_test.squeeze()
        data_test = torch.utils.data.TensorDataset(X_test, labels_test)
        loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=False)

        with torch.no_grad():
            iter_test = iter(loader_test)
            for i in range(len(loader_test)):
                base_model.eval()
                regression_model.eval()
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()

                if i == 0:    data_cum = inputs
                else:    data_cum = torch.cat((data_cum, inputs), 0)

                if i == 0:  all_label = labels.float()
                else:       all_label = torch.cat((all_label, labels.float()), 0)

                outputs = regression_model(base_model(inputs)).view(-1)

                if i == 0:
                    all_output[aug_way] = outputs.float().cpu()                          
                else:
                    all_output[aug_way] = torch.cat((all_output[aug_way], outputs.float().cpu()), 0)

                # update the mean and std
                if aug_way == 'identity':
                    base_model.train()
                    regression_model.train()
                    if (i + 1) >= test_batch and (i + 1) % test_batch == 0:
                        batch_test = data_cum[i - test_batch + 1: i + 1]
                        batch_test = batch_test.reshape(test_batch, 1, batch_test.shape[2], batch_test.shape[3])
                        batch_test = batch_test.cuda()
                        _ = base_model(batch_test)

    for aug_way in aug_dir:
        this_outputs = all_output[aug_way]   
        this_outputs = this_outputs.cuda()
        cc, rmse, mae = regression_metrics(all_label, this_outputs)

        print('BN-adapt: ' + 'CC = {:.4f}    RMSE = {:.4f}    MAE = {:.4f}'.format(cc, rmse, mae))


def test_augment(base_model, regression_model, tar_data, tar_label, args, test_batch=64):
    base_model = base_model.cuda()
    regression_model = regression_model.cuda()
    base_model.eval()
    regression_model.eval()

    aug_dir = [ "identity", "noise",
                "mult_0.9", "mult_1.1", "mult_1.2",
                "freq_high", "freq_low",
                "slide_1", "slide_2", "slide_3", "slide_4", "slide_5" ]
    all_output = {}
    PATH_TO_AUGED_DATA = '/PATH/TO/AUGED/DATA/'
    data_auged_dir = PATH_TO_AUGED_DATA + args.data + '/s' + str(args.testID)
    for aug_way in aug_dir:
        X_test, labels_test = load_aug(data_auged_dir, aug_way)

        labels_test = labels_test.squeeze()
        data_test = torch.utils.data.TensorDataset(X_test, labels_test)
        loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=False)

        with torch.no_grad():
            iter_test = iter(loader_test)
            for i in range(len(loader_test)):
                base_model.eval()
                regression_model.eval()
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()

                if i == 0:  all_label = labels.float()
                else:       all_label = torch.cat((all_label, labels.float()), 0)

                outputs = regression_model(base_model(inputs)).view(-1)

                if i == 0:
                    all_output[aug_way] = outputs.float().cpu()                          
                else:
                    all_output[aug_way] = torch.cat((all_output[aug_way], outputs.float().cpu()), 0)

    for aug_way in aug_dir:
        this_outputs = all_output[aug_way]   
        this_outputs = this_outputs.cuda()
        cc, rmse, mae = regression_metrics(all_label, this_outputs)

        if aug_way == aug_dir[1]:
            output_tensor = this_outputs.float().cpu().unsqueeze(0)
        elif aug_way != aug_dir[0]:
            output_tensor = torch.cat((output_tensor, this_outputs.float().cpu().unsqueeze(0)), dim=0)

        print(aug_way + ': ' + 'CC = {:.4f}    RMSE = {:.4f}    MAE = {:.4f}'.format(cc, rmse, mae))

    mean_output = output_tensor.mean(dim=0)
    mean_output = mean_output.cuda()
    cc, rmse, mae = regression_metrics(all_label, mean_output)

    print('Augment Avg: ' + 'CC = {:.4f}    RMSE = {:.4f}    MAE = {:.4f}'.format(cc, rmse, mae))
        
    
def test_augment_with_loss(model_loss, base_model, regression_model, tar_data, tar_label, args, test_batch=64):
    aug_names = [
        "identity", "noise",
        "mult_0.9", "mult_1.1", "mult_1.2",
        "freq_high", "freq_low",
        "slide_1", "slide_2", "slide_3", "slide_4", "slide_5"
    ]
    PATH_TO_AUGED_DATA = '/PATH/TO/AUGED/DATA/'
    data_root = PATH_TO_AUGED_DATA + args.data

    test_id = args.testID
    subject_ids = [test_id] 

    aug_dataset = AugmentedDataset(data_root, subject_ids, aug_names)
    loader_test = torch.utils.data.DataLoader(
        aug_dataset, batch_size=1, shuffle=False, drop_last=False
    )

    model_loss.eval()
    iter_test = iter(loader_test)
    with torch.no_grad():
        for i in range(len(loader_test)):
            base_model.eval()
            regression_model.eval()
            # [12][2]
            data = next(iter_test)
            trans_traininputs = []
            for j in range(len(data)):
                trans_traininputs.append((data[j][0].cuda(), data[j][1].cuda()))

            if i == 0:    data_cum = data[0][0]
            else:    data_cum = torch.cat((data_cum, data[0][0]), 0)

            x_aug_list = trans_traininputs
            labels = data[0][1].cuda()

            # get the reliability of different transformation
            pred_losses = []
            for j in range(len(x_aug_list)):
                x, _ = x_aug_list[j]
                x = base_model(x)
                pred_losses.append(model_loss(x))

            # Reliability weights w_{t,k}, Eq. (5); r predicts the task loss,
            # so the Softmax is taken over -r(z).
            pred_losses = torch.stack(pred_losses).squeeze()
            pred_losses = F.softmax(-pred_losses, dim=0)

            if i == 0:
                all_pred_losses = pred_losses.unsqueeze(0)
            else:
                all_pred_losses = torch.cat([all_pred_losses, pred_losses.unsqueeze(0)], dim=0)

            # Top-half selection of Eq. (8).  NOTE: the ranking is the running
            # average of w_{1..t,k} over the stream rather than the per-trial
            # ranking, and k = 6 is hard-coded instead of ceil(K/2).
            # calculate results based on reliability
            predicted_probs = all_pred_losses.mean(dim=0)
            topk_values, topk_indices = torch.topk(predicted_probs, k=6, largest=True)
            the_output = []
            for k in topk_indices:
                x, y = x_aug_list[k]
                target_output = regression_model(base_model(x)).view(-1)
                target_output = target_output.unsqueeze(0)
                the_output.append(target_output)
            mean_output = torch.mean(torch.stack(the_output), dim=0)

            if i == 0:
                all_output = mean_output.float().cpu()
                all_label = labels.float()
            else:
                all_output = torch.cat((all_output, mean_output.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

            # update the mean and std
            base_model.train()
            if (i + 1) >= test_batch and (i + 1) % test_batch == 0:
                batch_test = data_cum[i - test_batch + 1: i + 1]
                batch_test = batch_test.reshape(test_batch, 1, batch_test.shape[2], batch_test.shape[3])
                batch_test = batch_test.cuda()
                _ = base_model(batch_test)

        all_output = all_output.squeeze()
        all_output = all_output.cuda()
        all_label = all_label.cuda()

        cc, rmse, mae = regression_metrics(all_label.squeeze(), all_output)
        print('BFT-A: ' + 'CC = {:.4f}    RMSE = {:.4f}    MAE = {:.4f}'.format(cc, rmse, mae))
