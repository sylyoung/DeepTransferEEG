import torch.nn as nn
import torch


class Conv1(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=1, bias=False):
        super(Conv1, self).__init__()
        # TODO check this
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)
        x = self.conv1(x)
        x = self.bn(x)
        x = x.reshape(x.shape[0], -1)
        return x


class ConvChannel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=1, bias=False, groups=1):
        super(ConvChannel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, groups=groups)
        #self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        shape = x.shape[0]
        #print('in shape:', x.shape)
        x = self.conv1(x)
        #print('conv\'d shape:', x.shape)
        #x = self.bn(x)
        #print('bn\'d shape:', x.shape)
        x = x.reshape(shape, -1)
        #print('out shape:', x.shape)
        return x


class ConvFusion(nn.Module):
    def __init__(self, nn_deep, nn_out, in_channels=1, out_channels=1, kernel_size=1, bias=False, groups=1):
        super(ConvFusion, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, groups=groups)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels * 74)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, bias=False, groups=1)
        self.bn2 = nn.BatchNorm1d(num_features=1)
        self.fc = nn.Linear(nn_deep, nn_out)

    def forward(self, data):
        x_deep, x_feature = data
        shape = x_feature.shape[0]
        #print('in shape:', x_deep.shape, x_feature.shape)
        x_feature = self.conv1(x_feature)
        #print('conv1\'d shape:', x_feature.shape)
        x_feature = x_feature.reshape(shape, -1)
        x_feature = self.bn1(x_feature)
        #print('bn1\'d shape:', x_feature.shape)
        x = torch.cat((x_deep, x_feature), 1)
        #print('catted shape:', x.shape)
        x = x.unsqueeze_(1)
        x = self.conv2(x)
        #print('conv2\'d shape:', x.shape)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        #print('flatten\'d shape:', x.shape)
        x = self.fc(x)
        #print('out shape:', x.shape)
        return x


class ConvFeatureChannel(nn.Module):
    def __init__(self, nn_deep, nn_out, in_channels=1, out_channels=1, kernel_size=1, bias=False, groups=1):
        super(ConvFeatureChannel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, groups=groups)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels * 74)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, bias=False, groups=1)
        self.bn2 = nn.BatchNorm1d(num_features=1)
        self.fc = nn.Linear(nn_deep, nn_out)

    def forward(self, x):

        x = self.conv2(x)
        #print('conv2\'d shape:', x.shape)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        #print('flatten\'d shape:', x.shape)
        x = self.fc(x)
        #print('out shape:', x.shape)
        return x


class ConvChannelWiseFC(nn.Module):
    def __init__(self, nn_deep, nn_out, in_channels=1, out_channels=1, bias=False):
        super(ConvChannelWiseFC, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels * nn_deep)
        self.fc = nn.Linear(nn_deep * out_channels, nn_out)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.bn1(x)
        x = self.fc(x)
        return x


class ConvChannelWise(nn.Module):
    def __init__(self, nn_deep, nn_out, in_channels=1, out_channels=1, bias=False):
        super(ConvChannelWise, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels * nn_deep)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.bn1(x)
        return x