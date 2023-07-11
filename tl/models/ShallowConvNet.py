import torch
import torch.nn as nn


class ShallowConvNet(nn.Module):
    def __init__(self, n_classes, input_ch, fc_ch, batch_norm=True, batch_norm_alpha=0.1):
        super(ShallowConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 40

        if self.batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 13), stride=1, padding=(6, 7)),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5))

        self.fc = nn.Linear(fc_ch, n_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.layer1(x)
        x = torch.square(x)
        x = torch.nn.functional.avg_pool2d(x, (1, 35), (1, 7))
        x = torch.log(x)
        x = x.flatten(1)
        x = torch.nn.functional.dropout(x)
        x = self.fc(x)
        return x


class ShallowConvNet_feature(nn.Module):
    def __init__(self, n_classes, input_ch, fc_ch, batch_norm=True, batch_norm_alpha=0.1):
        super(ShallowConvNet_feature, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 40

        if self.batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 13), stride=1, padding=(6, 7)),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5))


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.layer1(x)
        x = torch.square(x)
        x = torch.nn.functional.avg_pool2d(x, (1, 35), (1, 7))
        x = torch.log(x)
        x = x.flatten(1)
        x = torch.nn.functional.dropout(x)
        return x