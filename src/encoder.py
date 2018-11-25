import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    """Recurrent neural network that encodes a given input sequence."""

    def __init__(self, src_vocab_size, embedding_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(src_vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, inputs, hidden_state):
        """
        inputs: [len]
        """
        inputs = inputs.view(-1, 1)
        embedded = self.embedding(inputs) # [len, 1, embedding_size]
        embedded = self.dropout(embedded)
        output, hidden_state = self.rnn(embedded, hidden_state)
        return output, hidden_state

    def init_hidden(self, device):
        hidden_state = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)
        return hidden_state
