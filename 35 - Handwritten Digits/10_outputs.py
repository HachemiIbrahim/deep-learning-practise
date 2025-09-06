import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

mnist_train = datasets.MNIST(root='./data', download=True, train=True, transform=ToTensor())
mnist_test = datasets.MNIST(root='./data', download=True, train=False, transform=ToTensor())

train_dataloader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for i in range(0, 10):
    loss_sum = 0
    for X, y in train_dataloader:
        X = X.reshape((-1, 784))
        y = F.one_hot(y, num_classes=10).type(torch.float32)
        print(y.shape)

        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        loss_sum+=loss.item()
        
    print(loss_sum)

#model.eval()
#with torch.no_grad():
#    accurate = 0
#    total = 0
#    for X, y in test_dataloader:
#        X = X.reshape((-1, 784))
#        y = (y == 0).reshape((-1, 1))
#
#        outputs = nn.functional.sigmoid(model(X))
#        correct_pred = ((outputs > 0.5) == y)
#        total+=correct_pred.size(0)
#
#        accurate+=correct_pred.type(torch.int).sum().item()
#    print(accurate / total)
