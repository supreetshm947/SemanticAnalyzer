import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from collections import Counter
from dataset import IMDBDataset
import pickle

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize


def save_vocab(vocab, path="models/vocab.pkl"):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(path="vocab.pkl"):
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def read_csv(path):
    return pd.read_csv(path)


def preprocess_data(df, test_size, random_state):
    # Remove line breaks
    df['review'] = df['review'].str.replace('<br />', ' ')
    # Remove special characters, just retain text and convert to lower case
    df['review'] = df['review'].str.replace('[^a-zA-Z]', '').str.lower()
    # Encoding labels as 0 or 1
    df['sentiment'] = LabelEncoder().fit_transform(df['sentiment'])

    return train_test_split(df['review'], df['sentiment'], test_size=test_size, random_state=random_state)


def get_loaders(path, test_size=0.1, random_state=41143, batch_size=64):
    df = read_csv(path)
    X_train, X_val, y_train, y_val = preprocess_data(df, test_size=test_size, random_state=random_state)

    train_tokens = [word_tokenize(review) for review in X_train]
    val_tokens = [word_tokenize(review) for review in X_val]

    counter = Counter([token for tokens in train_tokens for token in tokens])
    vocab = {word: idx for idx, (word, _) in enumerate(counter.most_common(10000))}

    save_vocab(vocab)

    train_numeric = [[vocab[token] for token in tokens if token in vocab] for tokens in train_tokens]
    val_numeric = [[vocab[token] for token in tokens if token in vocab] for tokens in val_tokens]

    train_padded = pad_sequence([torch.tensor(seq) for seq in train_numeric], batch_first=True)
    val_padded = pad_sequence([torch.tensor(seq) for seq in val_numeric], batch_first=True)

    train_labels = torch.tensor(y_train.values)
    val_labels = torch.tensor(y_val.values)

    train_dataset = IMDBDataset(train_padded, train_labels)
    val_dataset = IMDBDataset(val_padded, val_labels)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    return vocab, train_loader, val_loader
