import sys
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

# Torch: Creating X and y data (as tensors)
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])
y = torch.tensor(price, dtype=torch.float32)\
    .reshape((-1, 1))
y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std
# sys.exit()


model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0000000001)

for i in range(0, 1000):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    # print(loss)
    #if i % 100 == 0: 
    #    print(model.bias)
    #    print(model.weight)

prediction = model(torch.tensor([
    [5, 10000]
], dtype=torch.float32))
print(prediction * y_std + y_mean)