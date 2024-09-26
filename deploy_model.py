from data_ingestor import load_vocab
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
import torch
from sentiment_analyzer import SentimentAnalyser
import modelbit

mb = modelbit.login()


def load_model(path):
    # Load the model architecture arguments and state dict
    model_data = torch.load(path)

    vocab_size = model_data['vocab_size']
    embedding_dim = model_data['embedding_dim']
    hidden_dim = model_data['hidden_dim']
    output_dim = model_data['output_dim']

    model = SentimentAnalyser(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(model_data['state_dict'])

    return model


def compute_sentiment(review: str) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = load_vocab("models/vocab.pkl")
    new_tokens = word_tokenize(review)
    new_numeric = [vocab[token] for token in new_tokens if token in vocab]
    input = torch.tensor(new_numeric).to(device).unsqueeze(0)

    model = load_model("models/analyser.pth")
    model.to(device)
    output_numeric = model(input)
    result = "Positive" if output_numeric > 0.5 else "Negative"

    return result

mb.deploy(compute_sentiment)