from torch import nn
import torch


class ClippedLSTM(nn.Module):
    def __init__(self, *args, clip_forward=50, clip_backward=1000, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(*args, **kwargs)
        self.clip_forward = clip_forward
        self.clip_backward = clip_backward

        # Register hooks for backward clipping
        for name, param in self.lstm.named_parameters():
            if "weight" in name or "bias" in name:
                param.register_hook(
                    lambda grad: torch.clamp(grad, -clip_backward, clip_backward)
                )

    def forward(self, input, hidden=None):
        output, hidden = self.lstm(input, hidden)

        # Apply forward clipping to hidden states
        if isinstance(hidden, tuple):  # LSTM returns (h, c)
            h, c = hidden
            h = torch.clamp(h, -self.clip_forward, self.clip_forward)
            c = torch.clamp(c, -self.clip_forward, self.clip_forward)
            hidden = (h, c)

        # Apply forward clipping to output
        output = torch.clamp(output, -self.clip_forward, self.clip_forward)

        return output, hidden
