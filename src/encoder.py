import torch
import torch.nn as nn
from clipped_lstm import ClippedLSTM


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
        clip_forward=None,
        clip_backward=None,
    ):
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.src_vocab_size = src_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.clip_forward = clip_forward
        self.clip_backward = clip_backward

        self.embedding = nn.Embedding(src_vocab_size, embedding_size, padding_idx=2)
        self.dropout = nn.Dropout(dropout)
        self.rnn = ClippedLSTM(embedding_size, hidden_size, n_layers)

        # Learnable initial hidden state
        self.init_hidden_param = nn.Parameter(
            torch.randn(n_layers, 1, hidden_size) * 0.1
        )

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
        lengths = (inputs != 2).sum(dim=1).cpu().numpy().tolist()

        # inputs: [batch, len]
        embedded = self.embedding(inputs)  # [batch, len, embedding_size]
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(0, 1)  # [len, batch, embedding_size] for LSTM

        # 3. Pack sequences (PAD 무시)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=False, enforce_sorted=False
        )

        # 4. RNN 처리 (PAD가 hidden state에 영향 안줌)
        packed_output, hidden_state = self.rnn(packed_embedded, hidden_state)

        # if self.clip_forward is not None:
        # LSTM hidden_state is (hidden, cell) tuple
        # Apply clipping to both hidden state and cell state
        # hidden_State clip grad
        # hidden_state = (
        #     torch.clamp(
        #         hidden_state[0], min=-self.clip_forward, max=self.clip_forward
        #     ),
        #     torch.clamp(
        #         hidden_state[1], min=-self.clip_forward, max=self.clip_forward
        #     ),
        # )

        # output, hidden_state = self.rnn(
        #     embedded, hidden_state
        # )  # hidden_state is (h_n, c_n) tuple

        # 5. Unpack sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=False, padding_value=0.0
        )
        # output, (h_n, c_n) = super().forward(input, hx)

        return output, hidden_state

    def init_hidden(self, device, actual_batch_size=None):
        batch_size = (
            actual_batch_size if actual_batch_size is not None else self.batch_size
        )
        # LSTM requires both hidden state and cell state
        # Use learnable initial hidden state, expanded for batch size
        # -0.1, ~0.1
        hidden_state = nn.Parameter(
            torch.randn(self.n_layers, batch_size, self.hidden_size) * 0.1
        )
        hidden_state = hidden_state.to(device)

        # Initialize cell state to zeros for LSTM
        # cell_state = torch.zeros(
        #     self.n_layers, batch_size, self.hidden_size, device=device
        # )
        # hidden_state = hidden_state.to(device)
        # -0.1, 0.1 사이로 초기화
        cell_state = nn.Parameter(
            torch.randn(self.n_layers, batch_size, self.hidden_size) * 0.1,
        )
        cell_state = cell_state.to(device)

        return (hidden_state, cell_state)
