import torch
from torch import nn

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

print("Running on device:", device)

a = torch.tensor([5, 7], device=device)
b = torch.tensor([3, 4])

b = b.to(device)

print(a.device)
print(b.device)

print(a + b)
