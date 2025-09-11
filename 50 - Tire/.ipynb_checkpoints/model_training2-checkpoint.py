import sys
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

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
for param in resnet50_model.parameters():
    param.requires_grad = False
resnet50_model.eval()
resnet50_model = resnet50_model.to(device)

fc_model = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1)
)
fc_model = fc_model.to(device)

model = nn.Sequential(
    resnet50_model,
    fc_model
)
model = model.to(device)

optimizer = torch.optim.Adam(fc_model.parameters(), lr=0.00025)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(10):
    model.train()
    resnet50_model.eval()
    
    loss_sum = 0
    train_accurate = 0
    train_sum = 0
    for X, y in train_dataloader:
        X = X.to(device)
        y = y.to(device).type(torch.float).reshape(-1, 1)

        outputs = model(X)
        optimizer.zero_grad()
        loss = loss_fn(outputs, y)
        loss_sum+=loss.item()
        loss.backward()
        optimizer.step()

        predictions = torch.sigmoid(outputs) > 0.5
        accurate = (predictions == y).sum().item()
        train_accurate+=accurate
        train_sum+=y.size(0)
    print("Training loss: ", loss_sum / len(train_dataloader))
    print("Training accuracy: ", train_accurate / train_sum)


    model.eval()
    val_loss_sum = 0
    val_accurate = 0
    val_sum = 0
    with torch.no_grad():
        for X, y in val_dataloader:
            X = X.to(device)
            y = y.to(device).type(torch.float).reshape(-1, 1)

            outputs = model(X)
            loss = loss_fn(outputs, y)
            val_loss_sum+=loss.item()

            predictions = torch.sigmoid(outputs) > 0.5
            accurate = (predictions == y).sum().item()
            val_accurate+=accurate
            val_sum+=y.size(0)
    print("Validation loss: ", val_loss_sum / len(val_dataloader))
    print("Validation accuracy: ", val_accurate / val_sum)

