import sys
import torch
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
# print(dataset.class_to_idx)
# print(dataset[1400])

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
# print(len(train_dataset))
# print(len(val_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
