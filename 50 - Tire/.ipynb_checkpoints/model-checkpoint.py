import sys
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = torchvision.datasets.ImageFolder(
    root='./images',
    transform=preprocess
)
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

resnet50_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)
resnet50_model.fc = nn.Identity()
resnet50_model.eval()

for X, y in train_dataloader:
    print(resnet50_model(X).shape)
    print(resnet50_model(X))
    break

