import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
print("Running on device:", device)

mnist_test = datasets.FashionMNIST(root='./data', download=True, train=False, transform=ToTensor())

test_dataloader = DataLoader(mnist_test, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1, padding_mode="reflect"),
        nn.MaxPool2d(kernel_size=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Dropout(0.1)
    ),
    nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, padding_mode="reflect"),
        nn.MaxPool2d(kernel_size=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout(0.1)
    ),
    nn.Flatten(),
    nn.Sequential(
        nn.Linear(64 * 7 * 7, 1000),
        nn.BatchNorm1d(1000),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1000, 100),
        nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(100, 10)
    )
)
model.load_state_dict(torch.load('model_5.pth', weights_only=True, map_location=device))
model = model.to(device)

model.eval()
with torch.no_grad():
    accurate = 0
    total = 0
    for X, y in test_dataloader:
        X = X.to(device)
        y = y.to(device)
        outputs = nn.functional.softmax(model(X), dim=1) 
        correct_pred = (y == outputs.max(dim=1).indices)
        total+=correct_pred.size(0)
        accurate+=correct_pred.type(torch.int).sum().item()
    print("Accuracy on validation data:", accurate / total)