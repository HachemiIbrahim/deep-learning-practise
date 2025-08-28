from transformers import BartTokenizer, BartModel
import torch
from torch import nn

messages = [
    "We have release a new product, do you want to buy it?", 
    "Winner! Great deal, call us to get this product for free",
    "Tomorrow is my birthday, do you come to the party?",
]

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

out = tokenizer(messages, 
                padding=True, 
                max_length=512, 
                truncation=True, 
                return_tensors="pt")

bart_model = BartModel.from_pretrained("facebook/bart-base")
with torch.no_grad():
    bart_model.eval()
    print(out)

    pred = bart_model(
        input_ids=out["input_ids"], 
        attention_mask=out["attention_mask"]
    )
    embeddings = pred.last_hidden_state.mean(dim=1)
    print(embeddings.shape)
    print(embeddings[0, :])