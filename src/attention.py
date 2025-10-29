import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys


class Attention(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""

    def __init__(self, method, local=None, hidden_size=1000):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.local = local  # None, 'local-m', 'local-p'
        self.seq_len = 50

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
            self.location_layer = nn.Linear(self.hidden_size, self.seq_len, bias=False)
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
            # print("Using local attention mechanism.", file=sys.stderr)
            return self._local_attention(hidden, encoder_outputs, decoder_step)
        else:
            # print("Using global attention mechanism.", file=sys.stderr)
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
                batch_size, self.seq_len, hidden_size
            )
            concat_input = torch.cat((hidden_expanded, encoder_outputs_t), 2)
            position_weights = self.attention(
                concat_input
            )  # [batch_size, seq_len, hidden_size]
            energies = torch.einsum(
                "bsh,h->bs", position_weights, self.other.squeeze(0)
            )

        elif self.method == "location":
            # ✅ Dynamic location attention (no fixed max_seq_len)
            # Use linear layer to generate position-dependent weights
            # position_weights = self.location_layer(hidden)  # [batch_size, hidden_size]
            # energies = torch.einsum("bsh,bh->bs", encoder_outputs_t, position_weights)
            position_weights = self.location_layer(hidden)  # [batch_size, hidden_size]

        # Apply temperature scaling for numerical stability
        energies = energies

        return (
            F.softmax(energies, dim=1).unsqueeze(1),
            encoder_outputs,
        )  # [batch_size, 1, seq_len],

    def _local_attention(self, hidden, encoder_outputs, decoder_step):
        """Optimized vectorized local attention mechanism"""
        batch_size, hidden_size = hidden.size()
        seq_len, batch_size, _ = encoder_outputs.size()

        # ✅ Vectorized global attention computation first
        encoder_outputs_t = encoder_outputs.transpose(
            0, 1
        )  # [batch_size, seq_len, hidden_size]

        # Position Information
        positions = torch.arange(seq_len, dtype=torch.float, device=hidden.device) # []
        positions = positions.unsqueeze(0).expand(
            batch_size, -1
        )  # [batch_size, seq_len]
        src_position = None  # [batch_size]

        # initilaize mask
        window_mask = torch.ones(
            (batch_size, seq_len), device=hidden.device
        ).float()  # [batch_size, seq_len]

        if self.local == "local-m":
            # TODO: encoder_outputs_t 를 마스킹

            pt = torch.full(
                (batch_size,), decoder_step, dtype=torch.float, device=hidden.device
            )  # [batch_size]

            pt_expanded = pt.unsqueeze(1)  # [batch_size, 1]
            src_position = torch.floor(pt).long()  # [batch_size]
            window_mask = (
                torch.abs(positions - pt_expanded) <= self.window_size
            ).float()  # [batch_size, seq_len]

            encoder_outputs_t = encoder_outputs_t.masked_fill(
                window_mask.unsqueeze(2) == 0, 0.0
            )

        elif self.local == "local-p":
            # softmax
            tanh_output = torch.tanh(self.Wp(hidden))  # [batch_size, hidden_size]
            sigmoid_input = torch.matmul(tanh_output, self.vp).squeeze(
                -1
            )  # [batch_size]
            pt = seq_len * torch.sigmoid(sigmoid_input)  # [batch_size]
            # print(f"pt: {pt.shape}", file=sys.stderr)  # Debugging line
            pt_expanded = pt.unsqueeze(1)  # [batch_size, 1]
            # print(
            #     f"pt_expanded: {pt_expanded.shape}", file=sys.stderr
            # )  # Debugging line
            src_position = torch.floor(
                pt
            ).long()  # [batch_size] to [batch_size, seq_len]
            src_position = src_position.unsqueeze(1).expand(
                batch_size, seq_len
            )  # [batch_size, seq_len]

            # print(
            #     f"src_position: {src_position.shape}", file=sys.stderr
            # )  # Debugging line

            window_mask = (
                torch.abs(positions - pt_expanded) <= self.window_size
            ).float()  # [batch_size, seq_len]
            # print(
            #     f"window_mask: {window_mask.shape}", file=sys.stderr
            # )  # Debugging line
            # print(
            #     f"encoder_outputs_t before masking: {encoder_outputs_t.shape}",
            #     file=sys.stderr,
            # )  # Debugging line
            # encoder_outputs_t [batch_size, seq_len, hidden_size]
            encoder_outputs_t = encoder_outputs_t.masked_fill(
                window_mask.unsqueeze(2) == 0, 0.0
            )

        if self.method == "dot":
            # energies = torch.bmm(encoder_outputs_t, hidden.unsqueeze(2)).squeeze(2)
            energies = torch.einsum("bsh,bh->bs", encoder_outputs_t, hidden)
            # align = F.softmax(energies, dim=1)
        elif self.method == "general":
            # transformed = self.attention(encoder_outputs_t)
            # energies = torch.bmm(transformed, hidden.unsqueeze(2)).squeeze(2)
            transformed_hidden = self.attention(hidden)  # [batch_size, hidden_size]
            energies = torch.einsum(
                "bsh,bh->bs", encoder_outputs_t, transformed_hidden
            )  # [batch, seq_len, hidden_size] * [batch, hidden_size] -> [batch, seq_len]
            # align = F.softmax(energies, dim=1)
        elif self.method == "concat":
            hidden_expanded = hidden.unsqueeze(1).expand(
                batch_size, seq_len, hidden_size
            )
            concat_input = torch.cat((hidden_expanded, encoder_outputs_t), 2)
            energy = self.attention(concat_input)
            energies = torch.einsum("bsh,h->bs", energy, self.other.squeeze(0))
            # align = F.softmax(energies, dim=1)
        elif self.method == "location":
            energies = self.location_layer(hidden)  # [batch_size, hidden_size]
            # energies = torch.einsum("bsh,bh->bs", encoder_outputs_t, position_weights)
            # align = F.softmax(energies, dim=1)

        # Apply temperature scaling
        # energies = F.softmax(energies, dim=1)
        # energies = energies / math.sqrt(self.hidden_size)
        # attention_weights = F.softmax(energies, dim=1).unsqueeze(1)

        if self.local == "local-m":
            # 윈도우 내부는 1, 외부는 0
            # align_vector = align
            align_vector = F.softmax(energies, dim=1)
            return (
                align_vector.unsqueeze(1),
                encoder_outputs_t.transpose(0, 1),
            )  # ✅ [batch_size, 1, seq_len]

        elif self.local == "local-p":
            # Vectorized Gaussian calculation
            # 두가지 구현 방법 모두 존재
            # 1) gaussian_weight 더하고 softmax
            # 2) softmax 적용하고 gaussisna_weight (EMNLP 2015)
            D = self.window_size
            gaussian_weights = torch.exp(
                -((src_position - pt_expanded) ** 2) / (2 * (D / 2) ** 2)
            )  # [batch_size, seq_len]
            # align: [batch_size, seq_len]
            # gaussian_weights: [batch_size, seq_len]
            # align = F.softmax(energies, dim=1)
            align_vector = energies * gaussian_weights  # [batch_size, seq_len]
            align_vector = F.softmax(align_vector, dim=1)

            return (
                align_vector.unsqueeze(1),
                encoder_outputs_t.transpose(0, 1),
            )  # ✅ [batch_size, 1, seq_len]

            # Apply Gaussian weighting
            # energies = energies * gaussian_weights
            # energies = energies + torch.log(gaussian_weights + 1e-8)
            # attention_weights = F.softmax(energies, dim=1).unsqueeze(1)
            # attention_weights = attention_weights * gaussian_weights.unsqueeze(1)
        # attention_weights = F.softmax(energies, dim=1).unsqueeze(1)

        # ✅ Default fallback (should not reach here)
        return (
            F.softmax(
                torch.zeros(batch_size, seq_len, device=hidden.device), dim=1
            ).unsqueeze(1),
            encoder_outputs_t,
        )
