import torch 

X = torch.tensor([
    [10], 
    [38], 
    [100], 
    [150]
])
# print(X.size(1))
print(X[:, 0])