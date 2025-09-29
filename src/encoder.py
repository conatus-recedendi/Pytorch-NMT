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
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers)

        # Learnable initial hidden state
        self.init_hidden_param = nn.Parameter(torch.randn(n_layers, 1, hidden_size))

        # Initialize parameters with U[-0.1, 0.1]
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all parameters with uniform distribution U[-0.1, 0.1]"""
        for name, param in self.named_parameters():
            if param.dim() > 1:  # Weight matrices
                nn.init.uniform_(param, -0.1, 0.1)
            else:  # Bias vectors
                nn.init.uniform_(param, -0.1, 0.1)
        print(f"Encoder: Initialized all parameters with U[-0.1, 0.1]")

    def forward(self, inputs, hidden_state):
        """
        inputs: [batch, len]
        """
        # inputs: [batch, len]
        embedded = self.embedding(inputs)  # [batch, len, embedding_size]
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(0, 1)  # [len, batch, embedding_size] for LSTM
        output, hidden_state = self.rnn(
            embedded, hidden_state
        )  # hidden_state is (h_n, c_n) tuple
        return output, hidden_state

    def init_hidden(self, device, actual_batch_size=None):
        batch_size = (
            actual_batch_size if actual_batch_size is not None else self.batch_size
        )
        # LSTM requires both hidden state and cell state
        # Use learnable initial hidden state, expanded for batch size
        hidden_state = self.init_hidden_param.expand(
            self.n_layers, batch_size, self.hidden_size
        ).contiguous()

        # Initialize cell state to zeros for LSTM
        cell_state = torch.zeros(
            self.n_layers, batch_size, self.hidden_size, device=device
        )

        return (hidden_state, cell_state)
