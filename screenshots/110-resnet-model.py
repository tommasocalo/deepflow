import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        self.BatchNorm2d = torch.nn.BatchNorm2d(num_features=64, eps=0.00001, momentum=0.1, affine=True)
        self.ReLU = torch.nn.ReLU(inplace=False)
        self.Conv2d_2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        self.BatchNorm2d_2 = torch.nn.BatchNorm2d(num_features=128, eps=0.00001, momentum=0.1, affine=True)
        self.ReLU_2 = torch.nn.ReLU(inplace=False)
        self.Conv2d_3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        self.BatchNorm2d_3 = torch.nn.BatchNorm2d(num_features=256, eps=0.00001, momentum=0.1, affine=True)
        self.ReLU_3 = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        t1 = self.Conv2d(x)
        t2 = self.BatchNorm2d(t1)
        t3 = self.ReLU(t2)
        t4 = self.Conv2d_2(t3)
        t5 = self.BatchNorm2d_2(t4)
        t6 = self.ReLU_2(t5)
        t7 = self.Conv2d_3(t6)
        t8 = self.BatchNorm2d_3(t7)
        t9 = self.ReLU_3(t8)
        return t9
