import torch

weights = torch.load("fc_model_3.pth", weights_only=True, map_location="cpu")
torch.save(weights, "fc_model_3.pth")
print(weights)