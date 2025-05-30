import torch 
from torch import nn

X = torch.tensor([
    [10], 
    [38], 
    [100], 
    [150]
], dtype=torch.float32)

X = X.type(torch.int64)
print(X)
print(X.dtype)
# result = X * 0.5
# print(result)
# print(result.dtype)
