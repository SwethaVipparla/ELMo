import pandas as pd
import nltk
import unicodedata
import re

def preprocess(csv):
    df = pd.read_csv(csv)

    df = df.dropna()
    df = df.reset_index(drop=True)

    def normalize_unicode(s):
        return unicodedata.normalize('NFD', s)

    def preprocess_text(text):
        text = normalize_unicode(text)
        text = re.sub(r"(.)(\1{2,})", r"\1", text)
        text = re.sub(r"[^a-zA-Z.,!?]+", " ", text)
        text = text.strip().lower()
        return text

    nltk.download('punkt')

    df['Description'] = df['Description'].apply(preprocess_text)
    df['Description'] = df['Description'].apply(nltk.word_tokenize)
    
    return df

train_df = preprocess('train.csv')
test_df = preprocess('test.csv')