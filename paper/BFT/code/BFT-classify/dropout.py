# Deterministic dropout subnetwork bank for BFT-D, and the classification
# test-time loops built on it.  Equation numbers refer to the paper
# "Backpropagation-Free Test-Time Adaptation for Lightweight EEG-Based BCIs"
# (IEEE J. Biomed. Health Inform., 2026):
#
#   Eq. (2)  I^(k)_i = 0 for i in [(k-1)d/K, k d/K), 1 otherwise
#   Eq. (3)  z_t^(k) = 1/(1-p) * I^(k) . g(x_t)
#   Eq. (5)  w_{i,k} = softmax_k( r(z_i^(k)) )
#   Eq. (7)  y_t = argmax_c sum_k w_{t,k} softmax(h(z_t^(k))/tau)_c
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


# Monte-Carlo-dropout style baseline: the same K masked branches as BFT-D,
# averaged with equal weights instead of the reliability weights of Eq. (7).
# NOTE: no BatchNorm update is performed here, so this baseline runs without the
# BN-adapt that test_dropout_with_loss applies.
def test_dropout(block_model, classifier, X_test, labels_test, args):
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_test = X_test[:, :, :eeg_length]
    X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)
    X_test = X_test.unsqueeze(1)
    X_test, labels_test = X_test.cuda(), labels_test.cuda()
    
    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, 
                                            shuffle=False, drop_last=False)
    
    block_model.eval()
    classifier.eval()

    # The K masks of Eq. (2): mask k zeroes the contiguous feature block
    # [(k-1)d/K, k d/K) of g(x).  The bank is fixed, so branch k means the same
    # thing here as it did while the ranking module was trained.
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
            output1 = block_model(inputs)
            B, D = output1.shape
            for (start_r, end_r), key in zip(drop_ranges, range_keys):
                output1_mask = output1.clone()
                start = int(start_r * D)
                end = int(end_r * D)
                output1_mask[:, start:end] = 0.0

                outputs = classifier(output1_mask)

                if i == 0:
                    all_output[key] = outputs.float().cpu()                          
                else:
                    all_output[key] = torch.cat((all_output[key], outputs.float().cpu()), 0)
                    
    for key, value_list in all_output.items():
        value_list = nn.Softmax(dim=1)(value_list)
        if key == range_keys[0]:
            output_tensor = value_list.float().cpu().unsqueeze(0)
        else:
            output_tensor = torch.cat((output_tensor, value_list.float().cpu().unsqueeze(0)), dim=0)

    mean_output = output_tensor.mean(dim=0)
    _, predict = torch.max(mean_output, 1)
    pred = torch.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred) * 100
    print('Dropout avg test Acc = {:.2f}'.format(acc))


# BFT-D inference, i.e. the online test phase of Algorithm 1 for the
# deterministic dropout bank.  One trial at a time: mask the feature vector K
# ways by Eq. (2), score each branch with the ranking module r, and aggregate
# the sharpened Softmax outputs by Eq. (7).
def test_dropout_with_loss(model_loss, block_model, classifier, X_test, labels_test, args, test_batch=8):
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_test = X_test[:, :, :eeg_length]
    X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)
    X_test = X_test.unsqueeze(1)
    X_test, labels_test = X_test.cuda(), labels_test.cuda()
    
    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, 
                                            shuffle=False, drop_last=False)
    
    classifier.eval()
    model_loss.eval()

    num_splits = args.dropout_num
    drop_ranges = [(i / num_splits, (i + 1) / num_splits) for i in range(num_splits)]
    range_keys = [f"{start:.1f}-{end:.1f}" for start, end in drop_ranges]

    iter_test = iter(loader_test)
    with torch.no_grad():
        for i in range(len(loader_test)):
            block_model.eval()
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            if i == 0:    data_cum = inputs
            else:    data_cum = torch.cat((data_cum, inputs), 0)
            
            # {z_t^(k)} of Eq. (2).  NOTE: the 1/(1-p) rescaling of Eq. (3),
            # which keeps the expected feature magnitude unchanged under the
            # mask, is not applied here.  (BFT-regression instead divides the
            # head OUTPUT by (1-p), which also rescales the head bias.)
            # get the features after dropout
            pred_losses = []
            output1 = block_model(inputs)
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

            # Reliability weights w_{t,k}, Eq. (5).  model_loss predicts the
            # task loss, so a SMALL output means a reliable branch and the
            # Softmax is taken over -r(z); training uses the same negation.
            pred_losses = torch.stack(pred_losses).squeeze()
            pred_losses = F.softmax(-pred_losses, dim=0)

            all_mask = torch.stack(all_mask).squeeze()

            # NOTE: the weights actually used are the running average of
            # w_{1..t,k} over every trial seen so far, not the per-trial
            # w_{t,k} of Eq. (7).
            if i == 0:
                all_pred_losses = pred_losses.unsqueeze(0)
            else:
                all_pred_losses = torch.cat([all_pred_losses, pred_losses.unsqueeze(0)], dim=0)
            if i == 0:
                predicted_probs = all_pred_losses.mean(dim=0)
            else:
                predicted_probs = all_pred_losses.mean(dim=0)

            # Weighted convex combination of Eq. (7).  The divisor 0.25 is the
            # temperature tau; Section IV-B of the paper states tau = 0.5.
            # calculate weighted results based on reliability
            the_output = []
            for k in range(all_mask.shape[0]):
                x = all_mask[k]
                target_output = classifier(x)
                target_output = target_output / 0.25
                target_output = target_output.unsqueeze(0)
                the_output.append(nn.Softmax(dim=1)(target_output))
            mean_output = (torch.stack(the_output).squeeze(1) * predicted_probs.unsqueeze(1)) / predicted_probs.sum()
            mean_output = mean_output.sum(dim=0).unsqueeze(0)

            if i == 0:
                all_output = mean_output.float().cpu()
                all_label = labels.float()
            else:
                all_output = torch.cat((all_output, mean_output.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
            
            # BN-adapt update from the last `test_batch` trials of the stream.
            # update the mean and std
            block_model.train()
            if (i + 1) >= test_batch:
                batch_test = data_cum[i - test_batch + 1: i + 1]
                batch_test = batch_test.reshape(test_batch, 1, batch_test.shape[2], batch_test.shape[3])
                batch_test = batch_test.cuda()
                _ = block_model(batch_test)

    _, predict = torch.max(all_output, 1)
    pred = torch.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred) * 100
    print('Dropout with loss test Acc = {:.2f}'.format(acc))
    
    
