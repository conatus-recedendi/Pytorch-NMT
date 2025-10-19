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
            # Location attention: we'll compute Wa*ht dynamically based on seq_len
            # Store a parameter matrix that we'll slice based on actual sequence length
            self.max_seq_len = 50  # Maximum expected sequence length
            self.location_weights = nn.Parameter(
                torch.FloatTensor(self.max_seq_len, self.hidden_size)
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
        """Global attention mechanism"""
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
        elif self.method == "location":
            # Location-based attention: at = softmax(Wa * ht)
            # Wa: [seq_len, hidden_size], ht: [batch_size, hidden_size]
            # Result: [batch_size, seq_len]

            # Use the appropriate slice of location_weights based on actual seq_len
            Wa = self.location_weights[:seq_len, :]  # [seq_len, hidden_size]

            # Compute energies: batch matrix multiplication
            # hidden: [batch_size, hidden_size]
            # Wa^T: [hidden_size, seq_len]
            energies = torch.matmul(hidden, Wa.t())  # [batch_size, seq_len]
        elif self.method == "base":
            # Base attention: at = softmax(ht)
            # ht: [batch_size, hidden_size]
            # Result: [batch_size, seq_len] by repeating ht seq_len times

            # energies = (
            #     hidden.unsqueeze(1).expand(batch_size, seq_len, hidden_size).sum(dim=2)
            # )
            return torch.ones(batch_size, 1, seq_len, device=hidden.device)

        # Apply temperature scaling for numerical stability
        energies = energies / math.sqrt(self.hidden_size)

        return F.softmax(energies, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]

    def _local_attention(self, hidden, encoder_outputs, decoder_step):
        """Local attention mechanism with windowing"""
        batch_size, hidden_size = hidden.size()
        seq_len, batch_size, _ = encoder_outputs.size()

        # Determine aligned position pt
        if self.local == "local-m":
            # Monotonic alignment: pt = t
            if decoder_step is None:
                decoder_step = seq_len // 2  # fallback to middle
            pt = torch.tensor(decoder_step, dtype=torch.float, device=hidden.device)
            pt = pt.expand(batch_size)  # [batch_size]

        elif self.local == "local-p":
            # Predictive alignment: pt = S * sigmoid(vp^T tanh(Wp*ht))
            # hidden: [batch_size, hidden_size]
            tanh_output = torch.tanh(self.Wp(hidden))  # [batch_size, hidden_size]
            # vp: [hidden_size, 1]
            sigmoid_input = torch.matmul(tanh_output, self.vp).squeeze(
                -1
            )  # [batch_size]
            pt = seq_len * torch.sigmoid(sigmoid_input)  # [batch_size]

        # Create window around pt
        D = self.window_size  # Window half-size

        # Calculate window bounds for each batch
        window_start = torch.clamp(pt - D, 0, seq_len - 1).long()  # [batch_size]
        window_end = torch.clamp(pt + D + 1, 1, seq_len).long()  # [batch_size]

        # For simplicity, use a fixed window size and handle variable lengths
        max_window_size = 2 * D + 1

        # Initialize attention weights
        attention_weights = torch.zeros(
            batch_size, max_window_size, device=hidden.device
        )

        for b in range(batch_size):
            if self.local == "local-p":
                pt_val = int(pt[b].item())
            else:
                pt_val = pt

            # Calculate window boundaries with safety checks
            start = max(0, min(pt_val - self.window_size, seq_len - 1))
            end = min(seq_len, max(pt_val + self.window_size + 1, start + 1))
            window_len = end - start

            if window_len <= 0:
                continue

            # Extract window from encoder outputs
            window_encoder = encoder_outputs[
                start:end, b : b + 1, :
            ]  # [window_len, 1, hidden_size]
            window_hidden = hidden[b : b + 1, :]  # [1, hidden_size]

            # Compute attention energies for this window
            window_energies = self._compute_energies(
                window_hidden, window_encoder
            )  # [1, window_len]

            # Apply Gaussian distribution for local-p
            if self.local == "local-p":
                positions = torch.arange(
                    start, end, dtype=torch.float, device=hidden.device
                )
                gaussian_weights = torch.exp(
                    -((positions - pt[b]) ** 2) / (2 * (D / 2) ** 2)
                )
                window_energies = window_energies * gaussian_weights.unsqueeze(0)

            # Normalize within window
            window_attention = F.softmax(window_energies, dim=1)  # [1, window_len]

            # Place in full attention tensor
            # Ensure window_attention size matches window_len with safety checks
            window_attn_squeezed = window_attention.squeeze(0)
            actual_size = window_attn_squeezed.size(0)
            target_size = min(window_len, actual_size, seq_len - start)

            # Safely assign to attention weights
            if target_size > 0:
                attention_weights[b, 0, start : start + target_size] = (
                    window_attn_squeezed[:target_size]
                )

        return attention_weights.unsqueeze(1)  # [batch_size, 1, max_window_size]

    def _compute_energies(self, hidden, encoder_outputs):
        """Compute attention energies for a given hidden state and encoder outputs"""
        batch_size, hidden_size = hidden.size()
        seq_len, batch_size, _ = encoder_outputs.size()

        if self.method == "dot":
            encoder_outputs_t = encoder_outputs.transpose(0, 1)
            energies = torch.bmm(encoder_outputs_t, hidden.unsqueeze(2)).squeeze(2)

        elif self.method == "general":
            encoder_outputs_t = encoder_outputs.transpose(0, 1)
            transformed = self.attention(encoder_outputs_t)
            energies = torch.bmm(transformed, hidden.unsqueeze(2)).squeeze(2)

        elif self.method == "concat":
            hidden_expanded = hidden.unsqueeze(1).expand(
                batch_size, seq_len, hidden_size
            )
            encoder_outputs_t = encoder_outputs.transpose(0, 1)
            concat_input = torch.cat((hidden_expanded, encoder_outputs_t), 2)
            energy = self.attention(concat_input)
            energies = torch.bmm(
                energy,
                self.other.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2),
            ).squeeze(2)

        elif self.method == "location":
            Wa = self.location_weights[:seq_len, :]
            energies = torch.matmul(hidden, Wa.t())

        elif self.method == "base":
            energies = (
                hidden.unsqueeze(1).expand(batch_size, seq_len, hidden_size).sum(dim=2)
            )

        # Apply temperature scaling
        energies = energies / math.sqrt(self.hidden_size)

        return energies  # [batch_size, seq_len]
