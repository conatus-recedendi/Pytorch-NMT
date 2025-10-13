import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention


class AttentionDecoderRNN(nn.Module):
    """Recurrent neural network that makes use of gated recurrent units to translate encoded input using attention."""

    def __init__(
        self,
        batch_size,
        tgt_vocab_size,
        embedding_size,
        hidden_size,
        attn_model,
        n_layers=1,
        dropout=0.1,
        local=None,
    ):
        super(AttentionDecoderRNN, self).__init__()
        self.batch_size = batch_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attn_model = attn_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.local = local  # For local attention

        # Define layers
        self.embedding = nn.Embedding(tgt_vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout)
        if attn_model == "base":
            self.Wc = None
            self.Ws = nn.Linear(hidden_size, tgt_vocab_size)
        else:
            self.Wc = nn.Linear(hidden_size * 2, hidden_size, bias=True)
            self.Ws = nn.Linear(hidden_size, tgt_vocab_size, bias=True)

        # Choose attention model
        if attn_model is not None:
            self.attention = Attention(
                attn_model, hidden_size=hidden_size, local=self.local
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
        print(f"AttentionDecoder: Initialized all parameters with U[-0.1, 0.1]")

    def forward(self, input, decoder_context, hidden_state, encoder_outputs):

        # Run through RNN
        input = input.view(1, -1)
        embedded = self.embedding(input)  # [1, -1, embedding_size]
        embedded = self.dropout(embedded)

        # 현 시점에서 LSTM 호출
        # rnn_input = torch.cat(
        #     (embedded, decoder_context), 2
        # )  # [1, -1, embedding_size + hidden_size]
        rnn_output, hidden_state = self.lstm(
            embedded, hidden_state
        )  # rnn_output: [1, batch, hidden_size]

        # Calculate attention
        if self.attn_model == "base":
            # decoder context는 사용하지 않음
            context = torch.zeros(
                1, embedded.size(1), self.hidden_size, device=embedded.device
            )
            attention_weights = None
            # output = F.tanh(self.out(rnn_output), dim=2)
            h_tilde = rnn_output

        else:
            attention_weights = self.attention(rnn_output.squeeze(0), encoder_outputs)
            #  print(attention_weights.shape)
            # context is weight sum of attention weight and encoder_output
            context = torch.bmm(
                attention_weights, encoder_outputs.transpose(0, 1)
            )  # [batch_size, 1, hidden_size]

            context = context.transpose(0, 1)  # [1, -1, hidden_size]

            h_tilde = torch.tanh(
                self.Wc(torch.cat((rnn_output, context), 2))
            )  # [1, -1, hidden_size]
        logits = self.Ws(h_tilde).squeeze(0)  # [batch, tgt_vocab_size]
        log_prob = F.log_softmax(logits, dim=1)

        return log_prob, context, hidden_state, attention_weights
