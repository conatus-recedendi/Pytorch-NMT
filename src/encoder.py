import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    """Recurrent neural network that encodes a given input sequence."""

    def __init__(
        self,
        batch_size,
        src_vocab_size,
        embedding_size,
        hidden_size,
        n_layers=1,
        dropout=0.1,
    ):
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.src_vocab_size = src_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embedding_size, hidden_size, n_layers)

    def forward(self, inputs, hidden_state):
        """
        inputs: [batch, len]
        """
        # inputs: [batch, len]
        embedded = self.embedding(inputs)  # [batch, len, embedding_size]
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(0, 1)  # [len, batch, embedding_size] for GRU
        output, hidden_state = self.rnn(embedded, hidden_state)
        return output, hidden_state

    def init_hidden(self, device, actual_batch_size=None):
        batch_size = (
            actual_batch_size if actual_batch_size is not None else self.batch_size
        )
        hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(
            device
        )
        return hidden_state
