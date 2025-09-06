import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

from torchvision.transforms import ToTensor

from PIL import Image

model = nn.Sequential(
    nn.Linear(784, 50),
    nn.ReLU(),
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
model.load_state_dict(torch.load('mnist-model.pth', weights_only=True))

img = Image.open("4.png")
img.thumbnail((28, 28))
img = img.convert('L')

t = ToTensor()
X = t(img).reshape((-1, 784))
print(X.shape)
outputs = model(X)
print(nn.functional.softmax(outputs, dim=1))
