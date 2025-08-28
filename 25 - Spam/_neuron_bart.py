import sys
import torch
from torch import nn
import pandas as pd
from transformers import BartTokenizer, BartModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the data
df = pd.read_csv("./data/SMSSpamCollection", 
                 sep="\t", 
                 names=["type", "message"])

df["spam"] = (df["type"] == "spam").astype(int)
df.drop("type", axis=1, inplace=True)

# Split the data
df_train, df_val = train_test_split(df, test_size=0.2, random_state=0)

# Load BART and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model_bart = BartModel.from_pretrained("facebook/bart-base")

# Tokenize messages with a progress bar
def tokenize_texts(texts):
    return tokenizer(texts.tolist(), 
                     padding=True, 
                     truncation=True, 
                     return_tensors="pt", 
                     max_length=512)

# Transform training data with progress bar
print("Tokenizing and transforming training data...")
tokens_train = tokenize_texts(df_train["message"])
embeddings_train = []
for i in tqdm(range(len(tokens_train['input_ids']))):
    with torch.no_grad():
        output = model_bart(input_ids=tokens_train['input_ids'][i].unsqueeze(0), 
                            attention_mask=tokens_train['attention_mask'][i].unsqueeze(0))
        embeddings_train.append(output.last_hidden_state.mean(dim=1).squeeze(0))

# Transform validation data with progress bar
print("Tokenizing and transforming validation data...")
tokens_val = tokenize_texts(df_val["message"])
embeddings_val = []
for i in tqdm(range(len(tokens_val['input_ids']))):
    with torch.no_grad():
        output = model_bart(input_ids=tokens_val['input_ids'][i].unsqueeze(0), 
                            attention_mask=tokens_val['attention_mask'][i].unsqueeze(0))
        embeddings_val.append(output.last_hidden_state.mean(dim=1).squeeze(0))

# Stack the embeddings into tensors
X_train = torch.stack(embeddings_train)
y_train = torch.tensor(df_train["spam"].values, dtype=torch.float32).unsqueeze(1)

X_val = torch.stack(embeddings_val)
y_val = torch.tensor(df_val["spam"].values, dtype=torch.float32).unsqueeze(1)

# Define a simple linear model
model = nn.Linear(X_train.size(1), 1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

# Training loop
for i in range(10000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0: 
        print(f"Iteration {i}, Loss: {loss.item()}")

# Evaluation function
def evaluate_model(X, y): 
    model.eval()
    with torch.no_grad():
        y_pred = torch.sigmoid(model(X)) > 0.25
        accuracy = (y_pred == y).type(torch.float32).mean().item()
        sensitivity = (y_pred[y == 1] == y[y == 1]).type(torch.float32).mean().item()
        specificity = (y_pred[y == 0] == y[y == 0]).type(torch.float32).mean().item()
        precision = (y_pred[y_pred == 1] == y[y_pred == 1]).type(torch.float32).mean().item()
        
        print(f"Accuracy: {accuracy}")
        print(f"Sensitivity: {sensitivity}")
        print(f"Specificity: {specificity}")
        print(f"Precision: {precision}")

# Evaluate on training data
print("Evaluating on the training data")
evaluate_model(X_train, y_train)

# Evaluate on validation data
print("Evaluating on the validation data")
evaluate_model(X_val, y_val)

# Custom messages for prediction
custom_messages = pd.Series([
    "We have released a new product, do you want to buy it?", 
    "Winner! Great deal, call us to get this product for free",
    "Tomorrow is my birthday, do you want to come to the party?"
])

tokens_custom = tokenize_texts(custom_messages)
embeddings_custom = []
print("Transforming custom messages...")
for i in tqdm(range(len(tokens_custom['input_ids']))):
    with torch.no_grad():
        output = model_bart(input_ids=tokens_custom['input_ids'][i].unsqueeze(0), 
                            attention_mask=tokens_custom['attention_mask'][i].unsqueeze(0))
        embeddings_custom.append(output.last_hidden_state.mean(dim=1).squeeze(0))

X_custom = torch.stack(embeddings_custom)
with torch.no_grad():
    model.eval()
    pred = torch.sigmoid(model(X_custom))
    print(pred)
