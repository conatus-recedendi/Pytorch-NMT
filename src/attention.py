import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""

    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Define layers
        if self.method == "general":
            self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == "concat":
            self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

        # Initialize parameters with U[-0.1, 0.1]
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all parameters with uniform distribution U[-0.1, 0.1]"""
        for name, param in self.named_parameters():
            if param.dim() > 1:  # Weight matrices
                nn.init.uniform_(param, -0.1, 0.1)
            else:  # Bias vectors
                nn.init.uniform_(param, -0.1, 0.1)
        print(f"Attention: Initialized all parameters with U[-0.1, 0.1]")

    def forward(self, hidden, encoder_outputs):
        """Attend all encoder inputs conditioned on the previous hidden state of the decoder.

        Vectorized implementation for much better performance.

        Args:
            hidden: decoder hidden output used for condition  [batch_size, hidden_size]
            encoder_outputs: encoder outputs [seq_len, batch_size, hidden_size]

        Returns:
             Normalized (0..1) energy values, [batch_size, 1, seq_len]
        """
        batch_size, hidden_size = hidden.size()
        seq_len, batch_size, _ = encoder_outputs.size()

        # Vectorized attention computation
        if self.method == "dot":
            # hidden: [batch_size, hidden_size]
            # encoder_outputs: [seq_len, batch_size, hidden_size]
            # Transpose to [batch_size, seq_len, hidden_size]
            encoder_outputs_t = encoder_outputs.transpose(0, 1)
            # Compute dot product: [batch_size, seq_len]
            energies = torch.bmm(encoder_outputs_t, hidden.unsqueeze(2)).squeeze(2)

        elif self.method == "general":
            # Transform encoder outputs
            encoder_outputs_t = encoder_outputs.transpose(
                0, 1
            )  # [batch_size, seq_len, hidden_size]
            transformed = self.attention(
                encoder_outputs_t
            )  # [batch_size, seq_len, hidden_size]
            # Compute energies
            energies = torch.bmm(transformed, hidden.unsqueeze(2)).squeeze(2)

        elif self.method == "concat":
            # Repeat hidden for each time step
            hidden_expanded = hidden.unsqueeze(1).expand(
                batch_size, seq_len, hidden_size
            )
            encoder_outputs_t = encoder_outputs.transpose(0, 1)
            # Concatenate and compute
            concat_input = torch.cat((hidden_expanded, encoder_outputs_t), 2)
            energy = self.attention(concat_input)  # [batch_size, seq_len, hidden_size]
            energies = torch.bmm(
                energy,
                self.other.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2),
            ).squeeze(2)

        # Apply temperature scaling for numerical stability
        energies = energies / math.sqrt(self.hidden_size)

        return F.softmax(energies, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]
