import torch.nn as nn
import torch

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
