import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention


class AttentionDecoderRNN(nn.Module):
    """Recurrent neural network that makes use of gated recurrent units to translate encoded input using attention."""

    def __init__(self,
                 tgt_vocab_size,
                 embedding_size,
                 hidden_size,
                 attn_model,
                 n_layers=1,
                 dropout=.1):
        super(AttentionDecoderRNN, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attn_model = attn_model
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(tgt_vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, tgt_vocab_size)

        # Choose attention model
        if attn_model is not None:
            self.attention = Attention(attn_model, hidden_size)

    def forward(self, input, decoder_context, hidden_state, encoder_outputs):
        """Run forward propagation one step at a time.

        Get the embedding of the current input word (last output word) [s = 1 x batch_size x seq_len]
        then combine them with the previous context. Use this as input and run through the RNN. Next,
        calculate the attention from the current RNN state and all encoder outputs. The final output
        is the next word prediction using the RNN hidden_state state and context vector.

        Args:
            input: torch Variable representing the word input constituent
            decoder_context: torch Variable representing the previous context
            hidden_state: torch Variable representing the previous hidden_state state output
            encoder_outputs: torch Variable containing the encoder output values

        Return:
            output: torch Variable representing the predicted word constituent
            context: torch Variable representing the context value
            hidden_state: torch Variable representing the hidden_state state of the RNN
            attention_weights: torch Variable retrieved from the attention model
        """

        # Run through RNN
        input = input.view(1, -1)
        embedded = self.embedding(input) # [1, -1, embedding_size]
        embedded = self.dropout(embedded)

        #  print(embedded.shape)
        #  print(decoder_context.shape)
        rnn_input = torch.cat((embedded, decoder_context), 2) # [1, -1, embedding_size + hidden_size]
        rnn_output, hidden_state = self.gru(rnn_input, hidden_state) # [1, -1, hidden_size]

        # Calculate attention
        #  print(rnn_output.shape)
        #  print(encoder_outputs.shape)
        attention_weights = self.attention(rnn_output.squeeze(0), encoder_outputs)
        #  print(attention_weights.shape)
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1)) # [-1, 1, hidden_size]
        context = context.transpose(0, 1) # [1, -1, hidden_size]

        # Predict output
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 2)), dim=2)
        output = output.squeeze(0)
        return output, context, hidden_state, attention_weights
