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

model = nn.Sequential(
    nn.Conv2d(1, 3, kernel_size=(3, 3), padding=1, padding_mode="reflect"),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(2352, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

image = mnist_train[0][0].reshape(1, 1, 28, 28)
output = model(image)
print(output.shape)