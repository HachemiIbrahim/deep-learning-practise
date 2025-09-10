import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

mnist_train = datasets.FashionMNIST(root='./data', download=True, train=True, transform=ToTensor())
mnist_test = datasets.FashionMNIST(root='./data', download=True, train=False, transform=ToTensor())

train_dataloader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=32, shuffle=True)

