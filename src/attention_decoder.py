import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention
from clipped_lstm import ClippedLSTM

import sys


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
        clip_forward=None,
        clip_backward=None,
        input_forward=False,
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
        self.clip_forward = clip_forward
        self.clip_backward = clip_backward
        self.input_forward = input_forward

        # Define layers
        self.embedding = nn.Embedding(tgt_vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        if self.input_forward:
            print("Using input feeding in AttentionDecoderRNN.")
            print(self.input_forward)
            self.lstm = ClippedLSTM(
                embedding_size + hidden_size,
                hidden_size,
                n_layers,
                dropout=dropout,
                input_forward=True,
            )
        else:
            print("Using input feeding in AttentionDecoderRNN.")
            print(self.input_forward)
            self.lstm = ClippedLSTM(
                embedding_size,
                hidden_size,
                n_layers,
                dropout=dropout,
                input_forward=False,
            )
        if attn_model == "base":
            # self.Wc = nn.Linear(hidden_size * 2, hidden_size, bias=True)
            self.Wc = None  # Not used in 'base' attention
            self.Ws = nn.Linear(hidden_size, tgt_vocab_size, bias=True)
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

    def forward(
        self, input, decoder_context, hidden_state, encoder_outputs, decoder_step
    ):

        # Run through RNN
        input = input.view(1, -1)
        embedded = self.embedding(input)  # [1, -1, embedding_size]
        embedded = self.dropout(embedded)

        # 현 시점에서 LSTM 호출
        if self.input_forward:
            rnn_input = torch.cat(
                (embedded, decoder_context), 2
            )  # [1, -1, embedding_size + hidden_size]
        else:
            rnn_input = embedded
        # rnn_input = torch.cat(
        #     (embedded, decoder_context), 2
        # )  # [1, -1, embedding_size + hidden_size]
        rnn_output, hidden_state = self.lstm(
            rnn_input, hidden_state
        )  # rnn_output: [1, batch, hidden_size]

        if self.clip_forward is not None:
            # LSTM hidden_state is (hidden, cell) tuple
            # Apply clipping to both hidden state and cell state
            hidden_state = (
                torch.clamp(
                    hidden_state[0], min=-self.clip_forward, max=self.clip_forward
                ),
                torch.clamp(
                    hidden_state[1], min=-self.clip_forward, max=self.clip_forward
                ),
            )

        # Calculate attention
        if self.attn_model == "base":
            # Base model: no attention, no context
            output = F.log_softmax(self.Ws(rnn_output).squeeze(0), dim=1)

            # Return dummy context for interface consistency
            dummy_context = torch.zeros_like(decoder_context)
            attention_weights = None

            return output, dummy_context, hidden_state, attention_weights

        else:
            # Attention model
            attention_weights, encoder_outputs = self.attention(
                rnn_output.squeeze(0), encoder_outputs, decoder_step
            )  # [batch_size, 1, seq_len]
            assert (
                attention_weights.dim() == 3
            ), f"[ERROR] attention_weights should be 3D but got {attention_weights.dim()}D"
            assert (
                encoder_outputs.dim() == 3
            ), f"[ERROR] encoder_outputs should be 3D but got {encoder_outputs.dim()}D"
            assert (
                encoder_outputs.size(1) == 50
            ), f"[ERROR] encoder_outputs size(1) should be 50 but got {encoder_outputs.size(1)}"
            # context is weight sum of attention weight and encoder_output
            context = torch.bmm(
                attention_weights, encoder_outputs.transpose(0, 1)
            )  # [batch_size, 1, hidden_size]

            context = context.transpose(0, 1)  # [1, -1, hidden_size]

            # Attentional vector h̃_t = tanh(Wc[ht; ct])
            h_tilde = torch.tanh(
                self.Wc(torch.cat((rnn_output, context), 2))
            )  # [1, -1, hidden_size]
            logits = self.Ws(h_tilde).squeeze(0)  # [batch, tgt_vocab_size]
            log_prob = F.log_softmax(logits, dim=1)

            return log_prob, h_tilde, hidden_state, attention_weights
