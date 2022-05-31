from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from train_final import imshow, view_classify

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import argparse

trained_model = torch.load('full_model.pth')

test_transforms = transforms.Compose([transforms.Resize(225),
                                      transforms.CenterCrop(128),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])

