import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLikeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = torch.nn.Linear(in_features=128, out_features=64, bias=True)
        self.Linear_2 = torch.nn.Linear(in_features=64, out_features=32, bias=True)
        self.Linear_3 = torch.nn.Linear(in_features=32, out_features=16, bias=True)

    def forward(self, x):
        t1 = self.Linear(x)
        t2 = self.Linear_2(t1)
        t3 = self.Linear_3(t2)
        return t3
