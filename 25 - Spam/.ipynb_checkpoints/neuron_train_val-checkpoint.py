import sys
import torch
from torch import nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("./data/SMSSpamCollection", 
                 sep="\t", 
                 names=["type", "message"])

df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

df_train = df.sample(frac=0.8, random_state=0)
df_val = df.drop(index=df_train.index)

cv = CountVectorizer(max_features=5000)
messages_train = cv.fit_transform(df_train["message"])
messages_val = cv.transform(df_val["message"])

X_train = torch.tensor(messages_train.todense(), dtype=torch.float32)
y_train = torch.tensor(df_train["spam"].values, dtype=torch.float32)\
        .reshape((-1, 1))

X_val = torch.tensor(messages_val.todense(), dtype=torch.float32)
y_val = torch.tensor(df_val["spam"].values, dtype=torch.float32)\
        .reshape((-1, 1))

model = nn.Linear(5000, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

for i in range(0, 10000):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0: 
        print(loss)

def evaluate_model(X, y): 
    model.eval()
    with torch.no_grad():
        y_pred = nn.functional.sigmoid(model(X)) > 0.25
        print("accuracy:", (y_pred == y)\
            .type(torch.float32).mean())
        
        print("sensitivity:", (y_pred[y == 1] == y[y == 1])\
            .type(torch.float32).mean())
        
        print("specificity:", (y_pred[y == 0] == y[y == 0])\
            .type(torch.float32).mean())

        print("precision:", (y_pred[y_pred == 1] == y[y_pred == 1])\
            .type(torch.float32).mean()) 
        
print("Evaluating on the training data")
evaluate_model(X_train, y_train)

print("Evaluating on the validation data")
evaluate_model(X_val, y_val)

   