import torch
from torch import nn

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

print("Running on device:", device)

a = torch.tensor([5, 7], device=device, dtype=torch.float32)
b = torch.tensor([3, 4], device=device)

print(a + b)

model = nn.Sequential(
    nn.Linear(2, 10)
).to(device)

outputs = model(a.reshape(1, 2))
probabilities = nn.functional.softmax(outputs, dim=1)
print(probabilities)
# print(probabilities[0, 0].item())
print(probabilities[0, :].tolist())