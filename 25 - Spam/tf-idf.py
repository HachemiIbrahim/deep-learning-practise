import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("./data/SMSSpamCollection", 
                 sep="\t", 
                 names=["type", "message"])

df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

vectorizer = TfidfVectorizer(max_features=1000)
messages = vectorizer.fit_transform(df["message"])
print(messages[0, :])
print(vectorizer.get_feature_names_out()[888])