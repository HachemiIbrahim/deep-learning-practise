import sys
import os
import pandas as pd
import torch
from torch import nn

# Pandas: Reading the data
df = pd.read_csv("./data/used_cars.csv")

# Pandas: Preparing the data
age = df["model_year"].max() - df["model_year"]

milage = df["milage"]
milage = milage.str.replace(",", "")
milage = milage.str.replace(" mi.", "")
milage = milage.astype(int)

price = df["price"]
price = price.str.replace("$", "")
price = price.str.replace(",", "")
price = price.astype(int)

if not os.path.isdir("./model"):
    os.mkdir("./model")

# Torch: Creating X and y data (as tensors)
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
torch.save(X_mean, "./model/X_mean.pt")
torch.save(X_std, "./model/X_std.pt")
X = (X - X_mean) / X_std

y = torch.tensor(price, dtype=torch.float32)\
    .reshape((-1, 1))
y_mean = y.mean()
y_std = y.std()
torch.save(y_mean, "./model/y_mean.pt")
torch.save(y_std, "./model/y_std.pt")
y = (y - y_mean) / y_std
# sys.exit()


model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(0, 2500):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    #if i % 100 == 0: 
    #    print(loss)

torch.save(model.state_dict(), "./model/model.pt")