"""
LSTM model for time series classification
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Bidirectional LSTM classifier"""
    
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        
        self.input_size = config['model']['input_size']
        self.hidden_size = config['model']['hidden_size']
        self.num_layers = config['model']['num_layers']
        self.num_classes = config['model']['num_classes']
        self.dropout = config['model']['dropout']
        self.bidirectional = config['model'].get('bidirectional', True)
        
        self.num_directions = 2 if self.bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_size * self.num_directions, self.num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # LSTM forward
        out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Dropout
        hidden = self.dropout_layer(hidden)
        
        # Fully connected
        out = self.fc(hidden)
        
        return out
