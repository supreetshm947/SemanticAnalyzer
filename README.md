## 🌟 Semantic Analyzer 
- 🤖 Implementations of fully connected neural networks for sentiment analysis.
- 📈 Evaluation metrics including accuracy and loss.

## 📊 Data
The model is trained and evaluated on the IMDB movie review [Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). This dataset contains 50,000 reviews labeled as either positive or negative.

## 💻 Installation
To get started, clone the repository and install the required packages. You will need Python 3.10+ and pip.

```bash
git clone https://github.com/supreetshm947/SemanticAnalyzer
cd SemanticAnalyzer
make setup_env
```

## 🚀 Usage
### Data Preparation 
    ```
      make download_data
    ```
    Make sure your Kaggle Credentials are setup

### Running the Model

To train the model, run the following command:
    ```
      python train.py
    ```

## 🏗️ Model Architecture

The SentimentAnalyser class implements the sentiment analysis model. It consists of:
- An embedding layer 
- Two fully connected neural
- Activation functions such as ReLU and Sigmoid for output layer
  
Model
```
  class SentimentAnalyser(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentAnalyser, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)  # Map token indices to embedding vectors
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)
```
