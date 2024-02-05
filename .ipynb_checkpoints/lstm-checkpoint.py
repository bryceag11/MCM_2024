import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class LSTMMomentumPredictor(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, input_size):
        super(LSTMMomentumPredictor, self).__init__()
        self.hidden_dim = hidden_dim

        # Embedded layers for the player names
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer for sequence processing 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Output layer for binary classification
        self.hidden2output = nn.Linear(input_size, 1)

    def forward(self, sequence):
        embeds = self.word_embeddings(sequence)
        lstm_out, _ = self.lstm(embeds.view(len(sequence), 1, -1))
        output = self.hidden2output(lstm_out.view(len(sequence), -1))
        output = torch.sigmoid(output)
        return output

