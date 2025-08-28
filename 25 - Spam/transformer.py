import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("./data/SMSSpamCollection", 
                 sep="\t", 
                 names=["type", "message"])

df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

cv = CountVectorizer(max_features=1000)
messages = cv.fit_transform(df["message"])
print(messages[0, :])
print(cv.get_feature_names_out()[888])

from transformers import BertTokenizer, BertModel
import torch

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    r = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Get mean of last hidden states
    print(r.shape)
    return r

# Generate BERT embeddings for all messages
bert_features = np.array([get_bert_embedding(msg) for msg in df["message"]])
print(bert_features)