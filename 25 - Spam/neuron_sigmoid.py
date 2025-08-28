import torch
from torch import nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("./data/SMSSpamCollection", 
                 sep="\t", 
                 names=["type", "message"])

df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

cv = CountVectorizer(max_features=1000)
messages = cv.fit_transform(df["message"])

X = torch.tensor(messages.todense(), dtype=torch.float32)
y = torch.tensor(df["spam"], dtype=torch.float32)\
        .reshape((-1, 1))

model = nn.Linear(1000, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

for i in range(0, 10000):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0: 
        print(loss)

model.eval()
with torch.no_grad():
    y_pred = nn.functional.sigmoid(model(X))
    print(y_pred)
    print(y_pred.min())
    print(y_pred.max())