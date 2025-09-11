import sys
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from PIL import Image

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


resnet50_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)
resnet50_model.fc = nn.Identity()
resnet50_model = resnet50_model.to(device)

fc_model = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1)
)
fc_state_dict = torch.load("fc_model_3.pth", weights_only=True)
fc_model.load_state_dict(fc_state_dict)
fc_model = fc_model.to(device)

model = nn.Sequential(
    resnet50_model,
    fc_model
)
model = model.to(device)
model.eval()

tire = Image.open("./prediction/tire3.jpg")
tire_tensor = preprocess(tire)
tire_tensor = tire_tensor.unsqueeze(dim=0)
tire_tensor = tire_tensor.to(device)

with torch.no_grad():
    y_pred = torch.sigmoid(model(tire_tensor))
    print(y_pred)
    pass