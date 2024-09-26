from data_ingestor import get_loaders
from sentiment_analyzer import SentimentAnalyser
import torch
import torch.nn as nn
from tqdm import tqdm

EPOCHS = 1


def validate_model(model, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for review, label in tqdm(val_loader):
            review = review.to(device)
            label = label.to(device)
            output = model(review)
            loss = criterion(output.squeeze(), label.float())
            val_loss += loss.item()

            predicted = (output > 0.5).float()
            correct += (predicted == label.float()).sum().item()
            total += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for review, label in tqdm(train_loader):
            review = review.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(review)
            loss = criterion(output.squeeze(), label.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}')
        validate_model(model, val_loader)

def save_model(model, path, vocab_size, embedding_dim, hidden_dim, output_dim):
    model_data = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'state_dict': model.state_dict()
    }
    torch.save(model_data, path)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = "data/IMDB Dataset.csv"
    model_path = "models/analyser.pth"
    vocab, train_loader, val_loader = get_loaders(data_path)

    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 1

    # Initialize model with architecture parameters
    model = SentimentAnalyser(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

    # initializing params
    for param in model.parameters():
        if isinstance(param, nn.Linear):
            nn.init.xavier_uniform_(param.weight)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, criterion, optimizer, train_loader, val_loader, EPOCHS)

    save_model(model, model_path, vocab_size, embedding_dim, hidden_dim, output_dim)
