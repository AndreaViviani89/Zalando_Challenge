import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from collections import OrderedDict

# class Network(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.fc3 = nn.Linear(256, 128)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.fc4 = nn.Linear(128, 64)
#         self.bn4 = nn.BatchNorm1d(64)
#         self.out = nn.Linear(64, 10)
#         self.do = nn.Dropout(0.2, inplace=True)

#     def forward(self, x):
#         x = F.relu(self.do(self.bn1(self.fc1(x))))  
#         x = F.relu(self.do(self.bn2(self.fc2(x))))  
#         x = F.relu(self.do(self.bn3(self.fc3(x))))  
#         x = F.relu(self.do(self.bn4(self.fc4(x))))  
#         x = self.out(x)
#         x = F.softmax(x, dim=1)
#         return x


# model = Network()


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 10)
        self.do = nn.Dropout(0.2, inplace=True)

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(x.shape[0], -1)
        x = F.relu(self.do(self.bn1(self.fc1(x))))  
        x = F.relu(self.do(self.bn2(self.fc2(x))))  
        x = F.relu(self.do(self.bn3(self.fc3(x))))  
        x = F.relu(self.do(self.bn4(self.fc4(x))))  
        x = self.out(x)
        x = F.log_softmax(x, dim=1)
        return x

model2 = Network()

