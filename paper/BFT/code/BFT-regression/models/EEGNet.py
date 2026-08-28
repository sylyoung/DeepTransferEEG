import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score

import random
import time


# EEGNet-v4
class EEGNet(nn.Module):
    # default values:
    # kernLength = sample_rate / 2
    # F1 = 8, D = 2, F2 = 16
    # dropoutRate = 0.5 for within-subject, = 0.25 for cross-subject
    def __init__(self,
                 Chans: int,
                 Samples: int,
                 kernLength: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate:  float):
        super(EEGNet, self).__init__()

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

        self.regression_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                    out_features=1,
                    bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)
        output = self.regression_block(output)
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


class EEGNet_Regression(nn.Module):
    def __init__(self, regression_block):
        super().__init__()
        self.regression_block = regression_block

    def forward(self, x):
        return self.regression_block(x)
    