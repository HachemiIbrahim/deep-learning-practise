import sys
import torch
from torch import nn
import pandas as pd

df = pd.read_csv("./data/student_exam_data.csv")

X = torch.tensor(
    df[["Study Hours", "Previous Exam Score"]].values, 
    dtype=torch.float32
)
y = torch.tensor(df["Pass/Fail"], dtype=torch.float32)\
    .reshape((-1, 1))

model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

num_entries = X.size(0)
batch_size = 32

# print(X[32:64])
#for start in range(0, num_entries, batch_size):
#    end = min(num_entries, start + batch_size)
#    print("start: ", start, "end: ", end)

for i in range(0, 1000):
    loss_sum = 0
    for start in range(0, num_entries, batch_size):
        end = min(num_entries, start + batch_size)
        X_data = X[start:end]
        y_data = y[start:end]

        optimizer.zero_grad()
        outputs = model(X_data)
        loss = loss_fn(outputs, y_data)
        loss.backward()
        loss_sum += loss.item()
        optimizer.step()

    if i % 10 == 0:
        print(loss_sum)

model.eval()
with torch.no_grad():
    outputs = model(X)
    y_pred = nn.functional.sigmoid(outputs) > 0.5
    y_pred_correct = y_pred.type(torch.float32) == y
    print(y_pred_correct.type(torch.float32).mean())
