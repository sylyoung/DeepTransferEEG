# Deterministic dropout subnetwork bank for BFT-D on the regression tasks.
# Equation numbers refer to the paper "Backpropagation-Free Test-Time Adaptation
# for Lightweight EEG-Based BCIs" (IEEE J. Biomed. Health Inform., 2026):
#
#   Eq. (2)  I^(k)_i = 0 for i in [(k-1)d/K, k d/K), 1 otherwise
#   Eq. (3)  z_t^(k) = 1/(1-p) * I^(k) . g(x_t)
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


def test_dropout(base_model, regression_model, X_test, labels_test, args):
    base_model = base_model.cuda()
    regression_model = regression_model.cuda()
    base_model.eval()
    regression_model.eval()

    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate

    X_test = X_test[:, :, :eeg_length]
    X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.float32)
    X_test = X_test.unsqueeze(1)
    labels_test = labels_test.squeeze()
    X_test, labels_test = X_test.cuda(), labels_test.cuda()
    
    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=False)

    num_splits = args.dropout_num
    drop_ranges = [(i / num_splits, (i + 1) / num_splits) for i in range(num_splits)]
    range_keys = [f"{start:.1f}-{end:.1f}" for start, end in drop_ranges]
    all_output = {}
    
    with torch.no_grad():
        iter_test = iter(loader_test)
        for i in range(len(loader_test)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            if i == 0:  all_label = labels.float()
            else:       all_label = torch.cat((all_label, labels.float()), 0)

            # get the features after dropout
            output1 = base_model(inputs)
            B, D = output1.shape
            for (start_r, end_r), key in zip(drop_ranges, range_keys):
                output1_mask = output1.clone()
                start = int(start_r * D)
                end = int(end_r * D)
                output1_mask[:, start:end] = 0.0

                outputs = regression_model(output1_mask).view(-1)
                outputs = outputs / (1 - 1 / args.dropout_num)

                if i == 0:
                    all_output[key] = outputs.float().cpu()                          
                else:
                    all_output[key] = torch.cat((all_output[key], outputs.float().cpu()), 0)
                    
    for key, value_list in all_output.items():
        if key == range_keys[0]:    output_tensor = value_list.float().cpu().unsqueeze(0)
        else:   output_tensor = torch.cat((output_tensor, value_list.float().cpu().unsqueeze(0)), dim=0)

    mean_output = output_tensor.mean(dim=0)
    mean_output = mean_output.cuda()

    cc, rmse, mae = regression_metrics(all_label, mean_output)
    print('Dropout Avg: ' + 'CC = {:.4f}    RMSE = {:.4f}    MAE = {:.4f}'.format(cc, rmse, mae))


def test_dropout_with_loss(model_loss, base_model, regression_model, X_test, labels_test, args, test_batch=64):
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_test = X_test[:, :, :eeg_length]
    X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.float32)
    labels_test = labels_test.squeeze()
    X_test = X_test.unsqueeze(1)
    X_test, labels_test = X_test.cuda(), labels_test.cuda()
    
    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=False)
    
    regression_model.eval()
    model_loss.eval()

    num_splits = args.dropout_num
    drop_ranges = [(i / num_splits, (i + 1) / num_splits) for i in range(num_splits)]
    range_keys = [f"{start:.1f}-{end:.1f}" for start, end in drop_ranges]

    iter_test = iter(loader_test)
    with torch.no_grad():
        for i in range(len(loader_test)):
            base_model.eval()
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]

            if i == 0:    data_cum = inputs
            else:    data_cum = torch.cat((data_cum, inputs), 0)

            # get the features after dropout
            pred_losses = []
            output1 = base_model(inputs)
            B, D = output1.shape
            all_mask = []
            for (start_r, end_r), key in zip(drop_ranges, range_keys):
                output1_mask = output1.clone()
                start = int(start_r * D)
                end = int(end_r * D)
                output1_mask[:, start:end] = 0.0
                all_mask.append(output1_mask)

                # get the reliability of different dropout
                pred_losses.append(model_loss(output1_mask))
            
            # Reliability weights w_{t,k}, Eq. (5); r predicts the task loss,
            # so the Softmax is taken over -r(z).
            pred_losses = torch.stack(pred_losses).squeeze()
            pred_losses = F.softmax(-pred_losses, dim=0)

            # 10 * dim
            all_mask = torch.stack(all_mask).squeeze()

            if i == 0:
                all_pred_losses = pred_losses.unsqueeze(0)
            else:
                all_pred_losses = torch.cat([all_pred_losses, pred_losses.unsqueeze(0)], dim=0)

            # NOTE: the ranking used below is the running average of
            # w_{1..t,k} over the stream, not the per-trial ranking of Eq. (8).
            predicted_probs = all_pred_losses.mean(dim=0)

            # Top-half selection of Eq. (8).  k = 5 is hard-coded and equals
            # ceil(K/2) only at the default args.dropout_num = 10.
            # calculate results based on reliability
            topk_values, topk_indices = torch.topk(predicted_probs, k=5, largest=True)
            the_output = []
            for k in topk_indices:
                x = all_mask[k]
                target_output = regression_model(x).view(-1)
                # the 1/(1-p) factor of Eq. (3), applied to the head OUTPUT
                # rather than to the masked feature, so the head bias is
                # rescaled as well
                target_output = target_output / (1 - 1 / args.dropout_num)
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

        cc, rmse, mae = regression_metrics(all_label, all_output)
        print('Dropout Avg: ' + 'CC = {:.4f}    RMSE = {:.4f}    MAE = {:.4f}'.format(cc, rmse, mae))
