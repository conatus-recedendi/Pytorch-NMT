import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""

    def __init__(self, method, local=None, hidden_size=1000):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.local = local  # None, 'local-m', 'local-p'

        # Local attention window size (D in paper)
        self.window_size = 10  # 2*D+1 = 21 window

        # Define layers
        if self.method == "general":
            self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == "concat":
            self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))
        elif self.method == "location":
            # Location attention: use dynamic linear layer instead of fixed weights
            self.location_layer = nn.Linear(
                self.hidden_size, self.hidden_size, bias=False
            )
        elif self.method == "base":
            pass

        # Local attention predictive alignment parameters
        if self.local == "local-p":
            self.Wp = nn.Linear(self.hidden_size, self.hidden_size)
            self.vp = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))

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

    def forward(self, hidden, encoder_outputs, decoder_step=None):
        """Attend all encoder inputs conditioned on the previous hidden state of the decoder.

        Vectorized implementation for much better performance.

        Args:
            hidden: decoder hidden output used for condition  [batch_size, hidden_size]
            encoder_outputs: encoder outputs [seq_len, batch_size, hidden_size]
            decoder_step: current decoder step (for local-m alignment)

        Returns:
             Normalized (0..1) energy values, [batch_size, 1, seq_len] or [batch_size, 1, window_size]
        """
        batch_size, hidden_size = hidden.size()
        seq_len, batch_size, _ = encoder_outputs.size()

        # Check if using local attention
        if self.local is not None:
            return self._local_attention(hidden, encoder_outputs, decoder_step)
        else:
            return self._global_attention(hidden, encoder_outputs)

    def _global_attention(self, hidden, encoder_outputs):
        """Optimized global attention mechanism"""
        batch_size, hidden_size = hidden.size()
        seq_len, batch_size, _ = encoder_outputs.size()

        # ✅ Optimized vectorized attention computation
        encoder_outputs_t = encoder_outputs.transpose(
            0, 1
        )  # [batch_size, seq_len, hidden_size]

        if self.method == "dot":
            # ✅ Efficient einsum operation
            energies = torch.einsum("bsh,bh->bs", encoder_outputs_t, hidden)

        elif self.method == "general":
            # ✅ More efficient general attention with einsum
            # Transform hidden state once instead of encoder outputs
            transformed_hidden = self.attention(hidden)  # [batch_size, hidden_size]
            energies = torch.einsum("bsh,bh->bs", encoder_outputs_t, transformed_hidden)

        elif self.method == "concat":
            # ✅ Optimized concat attention
            hidden_expanded = hidden.unsqueeze(1).expand(
                batch_size, seq_len, hidden_size
            )
            concat_input = torch.cat((hidden_expanded, encoder_outputs_t), 2)
            energy = self.attention(concat_input)  # [batch_size, seq_len, hidden_size]
            energies = torch.einsum("bsh,h->bs", energy, self.other.squeeze(0))

        elif self.method == "location":
            # ✅ Dynamic location attention (no fixed max_seq_len)
            # Use linear layer to generate position-dependent weights
            if not hasattr(self, "location_layer"):
                self.location_layer = nn.Linear(
                    hidden_size, hidden_size, bias=False
                ).to(hidden.device)

            position_weights = self.location_layer(hidden)  # [batch_size, hidden_size]
            energies = torch.einsum("bsh,bh->bs", encoder_outputs_t, position_weights)

        elif self.method == "base":
            # Base attention: uniform distribution
            return torch.ones(batch_size, 1, seq_len, device=hidden.device) / seq_len

        # Apply temperature scaling for numerical stability
        energies = energies / math.sqrt(self.hidden_size)

        return F.softmax(energies, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]

    def _local_attention(self, hidden, encoder_outputs, decoder_step):
        """Optimized vectorized local attention mechanism"""
        batch_size, hidden_size = hidden.size()
        seq_len, batch_size, _ = encoder_outputs.size()

        # Determine aligned position pt
        if self.local == "local-m":
            # Monotonic alignment: pt = t (scalar for all batches)
            if decoder_step is None:
                decoder_step = seq_len // 2
            pt = torch.full(
                (batch_size,), decoder_step, dtype=torch.float, device=hidden.device
            )
        elif self.local == "local-p":
            # Predictive alignment: pt = S * sigmoid(vp^T tanh(Wp*ht))
            tanh_output = torch.tanh(self.Wp(hidden))  # [batch_size, hidden_size]
            sigmoid_input = torch.matmul(tanh_output, self.vp).squeeze(
                -1
            )  # [batch_size]
            pt = seq_len * torch.sigmoid(sigmoid_input)  # [batch_size]

        # ✅ Vectorized global attention computation first
        encoder_outputs_t = encoder_outputs.transpose(
            0, 1
        )  # [batch_size, seq_len, hidden_size]

        if self.method == "dot":
            energies = torch.bmm(encoder_outputs_t, hidden.unsqueeze(2)).squeeze(2)
        elif self.method == "general":
            transformed = self.attention(encoder_outputs_t)
            energies = torch.bmm(transformed, hidden.unsqueeze(2)).squeeze(2)
        elif self.method == "concat":
            hidden_expanded = hidden.unsqueeze(1).expand(
                batch_size, seq_len, hidden_size
            )
            concat_input = torch.cat((hidden_expanded, encoder_outputs_t), 2)
            energy = self.attention(concat_input)
            energies = torch.bmm(
                energy,
                self.other.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2),
            ).squeeze(2)
        else:
            # Fallback for other methods
            energies = torch.bmm(encoder_outputs_t, hidden.unsqueeze(2)).squeeze(2)

        # Apply temperature scaling
        energies = energies / math.sqrt(self.hidden_size)

        # ✅ Vectorized Gaussian weighting for local-p
        if self.local == "local-p":
            # Create position matrix [batch_size, seq_len]
            positions = torch.arange(seq_len, dtype=torch.float, device=hidden.device)
            positions = positions.unsqueeze(0).expand(
                batch_size, -1
            )  # [batch_size, seq_len]
            pt_expanded = pt.unsqueeze(1)  # [batch_size, 1]

            # Vectorized Gaussian calculation
            D = self.window_size
            gaussian_weights = torch.exp(
                -((positions - pt_expanded) ** 2) / (2 * (D / 2) ** 2)
            )  # [batch_size, seq_len]

            # Apply Gaussian weighting
            energies = energies * gaussian_weights

        # ✅ Vectorized windowing (optional - for memory efficiency)
        if self.window_size < seq_len:
            # Create window mask [batch_size, seq_len]
            positions = torch.arange(seq_len, dtype=torch.float, device=hidden.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            pt_expanded = pt.unsqueeze(1)

            # Window mask: 1 inside window, 0 outside
            window_mask = (
                torch.abs(positions - pt_expanded) <= self.window_size
            ).float()
            energies = (
                energies * window_mask + (window_mask - 1) * 1e9
            )  # -inf outside window

        # Final softmax
        attention_weights = F.softmax(energies, dim=1).unsqueeze(
            1
        )  # [batch_size, 1, seq_len]

        return attention_weights

    def _compute_energies(self, hidden, encoder_outputs):
        """Optimized compute attention energies for a given hidden state and encoder outputs"""
        batch_size, hidden_size = hidden.size()
        seq_len, batch_size, _ = encoder_outputs.size()

        # ✅ Use same optimized computations as _global_attention
        encoder_outputs_t = encoder_outputs.transpose(
            0, 1
        )  # [batch_size, seq_len, hidden_size]

        if self.method == "dot":
            energies = torch.einsum("bsh,bh->bs", encoder_outputs_t, hidden)

        elif self.method == "general":
            transformed_hidden = self.attention(hidden)  # [batch_size, hidden_size]
            energies = torch.einsum("bsh,bh->bs", encoder_outputs_t, transformed_hidden)

        elif self.method == "concat":
            hidden_expanded = hidden.unsqueeze(1).expand(
                batch_size, seq_len, hidden_size
            )
            concat_input = torch.cat((hidden_expanded, encoder_outputs_t), 2)
            energy = self.attention(concat_input)
            energies = torch.einsum("bsh,h->bs", energy, self.other.squeeze(0))

        elif self.method == "location":
            # Use dynamic location layer
            if not hasattr(self, "location_layer"):
                self.location_layer = nn.Linear(
                    hidden_size, hidden_size, bias=False
                ).to(hidden.device)
            position_weights = self.location_layer(hidden)
            energies = torch.einsum("bsh,bh->bs", encoder_outputs_t, position_weights)

        elif self.method == "base":
            energies = torch.ones(batch_size, seq_len, device=hidden.device)

        # Apply temperature scaling
        energies = energies / math.sqrt(self.hidden_size)

        return energies  # [batch_size, seq_len]
