import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LSTMMomentumPredictor(nn.Module):
    def __init__(self, hidden_layer_size=50, dropout_rate=0.5, sequence_length=302):
        super(LSTMMomentumPredictor, self).__init__()
        self.hidden_layer_size = hidden_layer_size 
        self.sequence_length = sequence_length

        # Convolutional layers 
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=16, hidden_size=self.hidden_layer_size, num_layers=1, batch_first=True, bidirectional=True)

        # Dropout 
        self.dropout = nn.Dropout(p=dropout_rate)

        # Fully connected layers for two outputs
        self.linear_p1 = nn.Linear(in_features=self.hidden_layer_size * 2, out_features=302)
        self.linear_p2 = nn.Linear(in_features=self.hidden_layer_size * 2, out_features=302)

    def forward(self, x):
        # Convolutional layers 
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # # LSTM transpose
        x = x.transpose(1, 2)

        # LSTM layer  
        lstm_out, _ = self.lstm(x)

        # Dropout application
        lstm_out = self.dropout(lstm_out)

        # Fully connected for two outputs
        predictions_p1 = F.sigmoid(self.linear_p1(lstm_out[:, -1, :]))
        predictions_p2 = F.sigmoid(self.linear_p2(lstm_out[:, -1, :]))

        return predictions_p1, predictions_p2

