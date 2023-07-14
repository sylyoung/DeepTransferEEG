# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : network.py
import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

from models.EEGNet import EEGNet_feature, EEGNet
from models.FC import FC, FC_xy


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


# dynamic change the weight of the domain-discriminator
def calc_coeff(iter_num, alpha=10.0, max_iter=10000.0):
    return float(2.0 / (1.0 + np.exp(-alpha * iter_num / max_iter)) - 1)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class Net_ln2(nn.Module):
    def __init__(self, n_feature, n_hidden, bottleneck_dim):
        super(Net_ln2, self).__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.ln1 = nn.LayerNorm(n_hidden)
        self.fc2 = nn.Linear(n_hidden, bottleneck_dim)
        self.fc2.apply(init_weights)
        self.ln2 = nn.LayerNorm(bottleneck_dim)

    def forward(self, x):
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        x = x.view(x.size(0), -1)
        return x


class Net_CFE(nn.Module):
    def __init__(self, input_dim=310, bottleneck_dim=64):
        if input_dim < 256:
            print('\nwarning', 'input_dim < 256')
        super(Net_CFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_dim, 256),
            # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, bottleneck_dim),  # default 64
            # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, hidden_dim, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(hidden_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(hidden_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class feat_classifier_xy(nn.Module):
    def __init__(self, class_num, bottleneck_dim, type="linear"):
        super(feat_classifier_xy, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        y = self.fc(x)
        return x, y


class scalar(nn.Module):
    def __init__(self, init_weights):
        super(scalar, self).__init__()
        self.w = nn.Parameter(tr.tensor(1.) * init_weights)

    def forward(self, x):
        x = self.w * tr.ones((x.shape[0]), 1).cuda()
        x = tr.sigmoid(x)
        return x


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class Discriminator(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ln1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = self.ln2(self.bn(x))
        y = tr.sigmoid(x)
        return y


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size1, hidden_size2):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size1)
        self.ad_layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.ad_layer3 = nn.Linear(hidden_size2, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]



