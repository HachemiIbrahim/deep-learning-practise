import torch 
from torch import nn

X = torch.tensor([
    [10], 
    [38], 
    [100], 
    [150]
], dtype=torch.float32)

model = nn.Linear(1, 1)

model.bias = nn.Parameter(
    torch.tensor([32], dtype=torch.float32)
)
model.weight = nn.Parameter(
    torch.tensor([[1.8]], dtype=torch.float32)
)

print(model.bias)
print(model.weight)

y_pred = model(X)
print(y_pred)