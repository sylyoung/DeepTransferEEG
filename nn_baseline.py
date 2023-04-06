import math
import os
import random
import time

import numpy as np
import torch
import sklearn
from torch.utils.data import TensorDataset, DataLoader

from models.FC import FC
from utils.alg_utils import soft_cross_entropy_loss
from utils.data_utils import split_data


def nn_fixepoch(
        model,
        learning_rate,
        num_iterations,
        metrics,
        cuda,
        cuda_device_id,
        seed,
        dataset,
        model_name,
        test_subj_id,
        label_probs,
        valid_percentage,
        train_x,
        train_y,
        test_x,
        test_y,
        loss_weights=None,
):
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:' + str(cuda_device_id))
        print('using cuda...')

    batch_size = 32 # 32 256 # TODO

    if valid_percentage != None and valid_percentage > 0:
        valid_indices = []
        for i in range(len(train_x) // 100):
            indices = np.arange(valid_percentage) + 100 * i
            valid_indices.append(indices)
        valid_indices = np.concatenate(valid_indices)
        valid_x = train_x[valid_indices]
        valid_y = train_y[valid_indices]
        train_x = np.delete(train_x, valid_indices, 0)
        train_y = np.delete(train_y, valid_indices, 0)

        print('train_x.shape, train_y.shape, valid_x.shape, valid_y.shape:', train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

    if label_probs:
        tensor_train_x, tensor_train_y = torch.from_numpy(train_x).to(
            torch.float32), torch.from_numpy(train_y).to(torch.float32)
    else:
        tensor_train_x, tensor_train_y = torch.from_numpy(train_x).to(
            torch.float32), torch.from_numpy(train_y.reshape(-1, )).to(torch.long)
    if model_name == 'EEGNet':
        tensor_train_x = tensor_train_x.unsqueeze_(3).permute(0, 3, 1, 2)
    train_dataset = TensorDataset(tensor_train_x, tensor_train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    if valid_percentage != None and valid_percentage > 0:
        if label_probs:
            tensor_valid_x, tensor_valid_y = torch.from_numpy(valid_x).to(
                torch.float32), torch.from_numpy(valid_y).to(torch.float32)
        else:
            tensor_valid_x, tensor_valid_y = torch.from_numpy(valid_x).to(
                torch.float32), torch.from_numpy(valid_y.reshape(-1,)).to(torch.long)
        if model_name == 'EEGNet':
            tensor_valid_x = tensor_valid_x.unsqueeze_(3).permute(0, 3, 1, 2)
        valid_dataset = TensorDataset(tensor_valid_x, tensor_valid_y)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    tensor_test_x, tensor_test_y = torch.from_numpy(test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    if model_name == 'EEGNet':
        tensor_test_x = tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset, batch_size=8)

    if cuda:
        model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if label_probs:
        criterion = soft_cross_entropy_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()
        if loss_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)
    if not os.path.isdir('./runs/'):
        os.mkdir('./runs/')
    if not os.path.isdir('./runs/' + str(dataset) + '/'):
        os.mkdir('./runs/' + str(dataset))

    # early stopping
    valid_cnter = 0
    best_valid_loss = 2 ** 20
    stop = False

    # Train the model
    for epoch in range(num_iterations):
        total_loss = 0
        cnt = 0
        for i, (x, y) in enumerate(train_loader):
            # Forward pass
            if cuda:
                x = x.cuda()
                y = y.cuda()

            outputs = model(x)

            loss = criterion(outputs, y)

            total_loss += loss
            cnt += 1

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        out_loss = total_loss / cnt

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_iterations, out_loss))
        """
        if (epoch + 1) % 10 == 0:
            model.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.cuda()
                    y = y.cuda()
                    outputs = model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    y_true.append(y.cpu())
                    y_pred.append(predicted.cpu())
                print('test score:', np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(
                    -1, ).tolist(), 5)[0])
            model.train()
        """
        if valid_percentage != None and valid_percentage > 0 and (epoch + 1) % 5 == 0:
            model.eval()
            total_loss_eval = 0
            cnt_eval = 0
            with torch.no_grad():
                for x, y in valid_loader:
                    if cuda:
                        x = x.cuda()
                        y = y.cuda()
                    outputs = model(x)
                    loss_eval = criterion(outputs, y)
                    total_loss_eval += loss_eval
                    cnt_eval += 1
                eval_loss = np.round(total_loss_eval.cpu().item() / cnt_eval, 5)
                print('valid loss:', eval_loss)

                if eval_loss < best_valid_loss:
                    best_valid_loss = eval_loss
                    valid_cnter = 0
                    torch.save(model, './runs/' + str(dataset) + '/' + model_name + '_testsubjID_' + str(test_subj_id) +
                               '_best.ckpt')
                else:
                    valid_cnter += 1
                if valid_cnter == 3:
                    stop = True
            model.train()

        if (epoch + 1) % 100 == 0:
            torch.save(model, './runs/' + str(dataset) + '/' + model_name + '_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '_seed_' + str(seed) + '.ckpt')
            print("--- %s seconds ---" % (time.time() - start_time))
        if stop:
            break

    '''
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tta = CoTTA(model, opt)

    y_true = []
    y_pred = []
    for x, y in test_loader:
        x = x.cuda()
        y = y.cuda()
        outputs = tta(x)
        _, predicted = torch.max(outputs.data, 1)
        y_true.append(y.cpu())
        y_pred.append(predicted.cpu())
    score = \
    np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, ).tolist(),
             5)[0]
    print('score:', score)
    # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
    return score
    '''


    # Test the model
    if valid_percentage != None and valid_percentage > 0:
        model = torch.load('./runs/' + str(dataset) + '/' + model_name + '_' + str(test_subj_id) + '_best.ckpt')
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in test_loader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            y_true.append(y.cpu())
            y_pred.append(predicted.cpu())
        score = np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, ).tolist(), 5)[0]
        print('score:', score)
        # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
        return score

