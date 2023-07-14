import mne
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import argparse
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier



import random
import sys
import os

from utils.alg_utils import EA


def apply_zscore(train_x, test_x, num_subjects):
    # train split into subjects
    train_z = []
    trial_num = int(train_x.shape[0] / (num_subjects - 1))
    for j in range(num_subjects - 1):
        scaler = preprocessing.StandardScaler()
        train_x_tmp = scaler.fit_transform(train_x[trial_num * j: trial_num * (j + 1), :])
        train_z.append(train_x_tmp)
    train_x = np.concatenate(train_z, axis=0)
    # test subject
    scaler = preprocessing.StandardScaler()
    test_x = scaler.fit_transform(test_x)
    return train_x, test_x


def data_loader(dataset):
    '''
    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''
    mne.set_log_level('warning')

    if dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))
        indices = np.concatenate(indices, axis=0)

        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def data_alignment(X, num_subjects):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X


def traintest_split_multisource(dataset, X, y, num_subjects, test_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    test_x = data_subjects.pop(test_subject_id)
    test_y = labels_subjects.pop(test_subject_id)
    train_x = data_subjects
    train_y = labels_subjects
    print('Test subject s' + str(test_subject_id))
    print('Training/Test split:', len(train_x), 'Source Subjects of', train_x[0].shape, test_x[0].shape)
    return train_x, train_y, test_x, test_y


def ml_classifier(approach, output_probability, train_x, train_y, test_x, return_model=None, weight=None):
    if approach == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif approach == 'LR':
        clf = LogisticRegression(max_iter=1000)
    elif approach == 'AdaBoost':
        clf = AdaBoostClassifier()
    elif approach == 'GradientBoosting':
        clf = GradientBoostingClassifier()
    elif approach == 'xgb':
        clf = XGBClassifier()
        if weight:
            print('XGB weight:', weight)
            clf = XGBClassifier(scale_pos_weight=weight)
            # clf = imb_xgb(special_objective='focal', focal_gamma=2.0)
    # clf = LinearDiscriminantAnalysis()
    # clf = SVC()
    # clf = LinearSVC()
    # clf = KNeighborsClassifier()
    clf.fit(train_x, train_y)

    if output_probability:
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    if return_model:
        return pred, clf
    else:
        print(pred)
        return pred


def convert_label(labels, axis, threshold, minus1=False):
    if minus1:
        # Converting labels to -1 or 1, based on a certain threshold
        if np.random.randint(2, size=1)[0] == 1:
            label_01 = np.where(labels > threshold, 1, -1)
        else:
            label_01 = np.where(labels >= threshold, 1, -1)
    else:
        # Converting labels to 0 or 1, based on a certain threshold
        if np.random.randint(2, size=1)[0] == 1:
            label_01 = np.where(labels > threshold, 1, 0)
        else:
            label_01 = np.where(labels >= threshold, 1, 0)
    return label_01



def SML_soft(preds):
    """
    Parameters
    ----------
    preds : numpy array
        data of shape (num_models, num_test_samples)

    Returns
    ----------
    pred : numpy array
        data of shape (num_test_samples)
    """
    soft = torch.from_numpy(preds).to(torch.float32)
    out = torch.mm(soft, soft.T)
    w, v = np.linalg.eig(out)
    accuracies = v[:, 0]
    total = np.sum(accuracies)
    weights = accuracies / total
    prediction = np.dot(weights, soft.numpy())
    pred = convert_label(prediction, 0, 0.5)
    return pred


def SML_soft_multiclass(preds):
    """
    Parameters
    ----------
    preds : numpy array
        data of shape (num_models, num_test_samples, num_classes)

    Returns
    ----------
    pred : numpy array
        data of shape (num_test_samples)
    """
    predictions = []
    class_num = preds.shape[-1]
    for i in range(class_num):
        soft = torch.from_numpy(preds[:, :, i]).to(torch.float32)
        out = torch.mm(soft, soft.T)
        w, v = np.linalg.eig(out)
        accuracies = v[:, 0]
        total = np.sum(accuracies)
        weights = accuracies / total
        prediction = np.dot(weights, soft.numpy())
        predictions.append(prediction)
    predictions = np.array(predictions)
    pred = np.argmax(predictions, axis=0)
    return pred




def ml_multisource(dataset, info, align, approach, combine_strategy, cuda_device_id, args):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(dataset)
    print('X, y, num_subjects, paradigm, sample_rate:', X.shape, y.shape, num_subjects, paradigm, sample_rate)

    print('sample rate:', sample_rate)

    if align:
        X = data_alignment(X, num_subjects)

    scores_arr = []

    for i in range(num_subjects):
        train_x, train_y, test_x, test_y = traintest_split_multisource(dataset, X, y, num_subjects, i)
        print('num of train_x, train_x, train_y, test_x, test_y.shape', len(train_x), train_x[0].shape, train_y[0].shape, test_x.shape, test_y.shape)

        if paradigm == 'MI':
            # NNM
            if combine_strategy == 'NNM':
                args.idt = i
                similarity_weights = NeuralNetworkMeasurement(test_x, args)

            # CSP
            subj_preds = []
            for s in range(len(train_x)):
                subj_train_x, subj_train_y = train_x[s], train_y[s]
                subj_csp = mne.decoding.CSP(n_components=10)
                subj_train_x_csp = subj_csp.fit_transform(subj_train_x, subj_train_y)
                subj_test_x_csp = subj_csp.transform(test_x)

                # classifier
                subj_pred, subj_model = ml_classifier(approach, True, subj_train_x_csp, subj_train_y, subj_test_x_csp, return_model=True)
                subj_preds.append(subj_pred)


            subj_preds = np.stack(subj_preds)
            print(subj_preds.shape)
            print(similarity_weights.shape)
            input('')

            # SML
            if dataset == 'BNCI2014001-4':
                pred = SML_soft_multiclass(subj_preds[:, :])
            else:
                pred = SML_soft(subj_preds[:, :, 1])

            # averaging
            #avg_pred = np.average(subj_preds, axis=0)
            #pred = np.argmax(avg_pred, axis=1)

            subj_scores = np.round(accuracy_score(test_y, pred), 5)
            score = np.mean(subj_scores)
        scores_arr.append(score)
    print('#' * 30)
    for i in range(len(scores_arr)):
        scores_arr[i] = np.round(scores_arr[i] * 100)
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))

    return scores_arr


class EEGNet_feature(nn.Module):

    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate:  float,
                 norm_rate: float):
        super(EEGNet_feature, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
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
            # SeparableConv2d
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)
        return output


class FC(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC, self).__init__()
        self.fc = nn.Linear(nn_in, nn_out)

    def forward(self, x):
        x = self.fc(x)
        return x


class FC_xy(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC_xy, self).__init__()
        self.nn_out = nn_out
        self.fc = nn.Linear(nn_in, nn_out)

    def forward(self, x):
        y = self.fc(x)
        return x, y


def backbone_net(args, return_type='y'):
    netF = EEGNet_feature(n_classes=args.class_num,
                        Chans=args.chn,
                        Samples=args.time_sample_num,
                        kernLenght=int(args.sample_rate // 2),
                        F1=4,
                        D=2,
                        F2=8,
                        dropoutRate=0.25,
                        norm_rate=0.5)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC


def NeuralNetworkMeasurement(test_x, args):
    '''

    Parameters
    ----------
    test_x: test data of numpy array (num_samples, num_channels, num_timesamples)

    Returns
    -------
    weights: weights of source domains of numpy array (num_samples, num_source_domains)
    '''

    args.class_num = args.N - 1
    print(args.class_num)
    netF, netC = backbone_net(args, return_type='xy')
    if args.device != torch.device('cpu'):
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)
    if args.align:
        if args.device != torch.device('cpu'):
            base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + 'Domain_Classifier' + '_' + str(args.backbone) +
                                                    '_S' + str(args.idt) + '_seed' + str(args.SEED) + '.ckpt'))
        else:
            base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + 'Domain_Classifier' + '_' + str(args.backbone) +
                                                    '_S' + str(args.idt) + '_seed' + str(args.SEED) + '.ckpt', map_location=torch.device('cpu')))

    test_x = torch.from_numpy(test_x).to(torch.float32)
    test_x = test_x.unsqueeze_(3)
    # EEGNet
    test_x = test_x.permute(0, 3, 1, 2)
    data_test = Data.TensorDataset(test_x)
    data_loader = Data.DataLoader(data_test, batch_size=32, shuffle=False, drop_last=False)
    with torch.no_grad():
        all_output = []
        for [x] in data_loader:
            if args.device != torch.device('cpu'):
                x = x.cuda()
            _, outputs = base_network(x)
            all_output.append(outputs)
    all_output = torch.nn.Softmax(dim=1)(torch.cat(all_output))
    all_output = all_output.detach().cpu().numpy()

    return all_output


if __name__ == '__main__':

    # cuda_device_id as args[1]
    if len(sys.argv) > 1:
        cuda_device_id = str(sys.argv[1])
    else:
        cuda_device_id = -1
    try:
        device = torch.device('cuda:' + cuda_device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_id
        print('using GPU')
    except:
        device = torch.device('cpu')
        print('using CPU (no CUDA)')

    scores = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    dataset_arr = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014001-4']

    combine_strategy = 'NNM'

    align = True

    for dataset in dataset_arr:

        if dataset == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if dataset == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if dataset == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        if dataset == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248

        for s in [1, 2, 3, 4, 5]:

            args = argparse.Namespace(feature_deep_dim=feature_deep_dim, trial_num=trial_num, device=device,
                                      time_sample_num=time_sample_num, sample_rate=sample_rate,
                                      N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=dataset)

            args.SEED = s
            args.align = align
            args.backbone = 'EEGNet'

            for approach in ['LDA']:

                print(dataset, align, approach)

                # info = dataset_to_file(dataset, data_save=False)

                ml_multisource(dataset, None, align, approach, combine_strategy, cuda_device_id, args)
