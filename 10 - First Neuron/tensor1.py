import torch 

b = torch.tensor(32)
w1 = torch.tensor(1.8)

X1 = torch.tensor([10, 38, 100, 150])

y_pred = 1 * b + X1 * w1
print(y_pred)