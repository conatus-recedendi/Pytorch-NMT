from torch import nn
import torch


class ClippedLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        clip_forward=50,
        clip_backward=50,
        input_forward=False,
        dropout=0,
        batch_first=False,
        **kwargs
    ):
        super().__init__()

        self.clip_forward = clip_forward
        self.clip_backward = clip_backward
        self.input_forward = input_forward
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if input_forward and num_layers > 1:
            # Multi-layer LSTM with different input sizes
            self.lstm_layers = nn.ModuleList()

            # First layer: input size = 2 * hidden_size
            first_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                dropout=0,  # No dropout for single layer
                batch_first=batch_first,
                **kwargs
            )
            self.lstm_layers.append(first_layer)

            # Subsequent layers: input size = hidden_size
            for i in range(1, num_layers):
                layer = nn.LSTM(
                    input_size=hidden_size,  # n (not 2n)
                    hidden_size=hidden_size,
                    num_layers=1,
                    dropout=0,  # No dropout for single layer
                    batch_first=batch_first,
                    **kwargs
                )
                self.lstm_layers.append(layer)

            # Apply dropout manually between layers if needed
            self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        else:
            print("Using standard LSTM without input feeding.")
            print(input_size, hidden_size, num_layers)
            # Standard LSTM or single layer
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=batch_first,
                **kwargs
            )

        # Register hooks for backward clipping
        # for name, param in self.named_parameters():
        #     if "weight" in name or "bias" in name:
        #         param.register_hook(
        #             lambda grad: torch.clamp(
        #                 grad, -self.clip_backward, self.clip_backward
        #             )
        # )

    def forward(self, input, hidden=None):
        if self.input_forward and self.num_layers > 1:
            # Multi-layer forward with different input sizes
            output = input

            if hidden is None:
                # Initialize hidden states for each layer
                batch_size = (
                    input.size(1)
                    if not self.lstm_layers[0].batch_first
                    else input.size(0)
                )
                hidden_states = []
                for _ in range(self.num_layers):
                    h = torch.zeros(
                        1, batch_size, self.hidden_size, device=input.device
                    )
                    c = torch.zeros(
                        1, batch_size, self.hidden_size, device=input.device
                    )
                    hidden_states.append((h, c))
            else:
                # Split hidden state for each layer
                if isinstance(hidden[0], tuple):
                    # Already separated by layers
                    hidden_states = hidden
                else:
                    # Split by layers
                    h_all, c_all = hidden
                    hidden_states = []
                    for i in range(self.num_layers):
                        h_layer = h_all[i : i + 1]  # [1, batch, hidden_size]
                        c_layer = c_all[i : i + 1]  # [1, batch, hidden_size]
                        hidden_states.append((h_layer, c_layer))

            new_hidden_states = []

            # Forward through each layer
            for i, (lstm_layer, layer_hidden) in enumerate(
                zip(self.lstm_layers, hidden_states)
            ):
                output, new_layer_hidden = lstm_layer(output, layer_hidden)

                # Apply forward clipping
                h, c = new_layer_hidden
                # h = torch.clamp(h, -self.clip_forward, self.clip_forward)
                # c = torch.clamp(c, -self.clip_forward, self.clip_forward)
                new_layer_hidden = (h, c)
                new_hidden_states.append(new_layer_hidden)

                # Apply dropout between layers (except last layer)
                if i < len(self.lstm_layers) - 1 and self.dropout is not None:
                    output = self.dropout(output)

            # Combine hidden states from all layers
            all_h = torch.cat([h for h, c in new_hidden_states], dim=0)
            all_c = torch.cat([c for h, c in new_hidden_states], dim=0)
            final_hidden = (all_h, all_c)

        else:
            # Standard LSTM forward
            output, final_hidden = self.lstm(input, hidden)

            # Apply forward clipping to hidden states
            if isinstance(final_hidden, tuple):  # LSTM returns (h, c)
                h, c = final_hidden
                # h = torch.clamp(h, -self.clip_forward, self.clip_forward)
                # c = torch.clamp(c, -self.clip_forward, self.clip_forward)
                final_hidden = (h, c)

        # Apply forward clipping to output
        # output = torch.clamp(output, -self.clip_forward, self.clip_forward)

        # Apply gradient norm clipping during training
        # if self.training:
        #     torch.nn.utils.clip_grad_norm_(
        #         self.parameters(), max_norm=self.clip_backward
        # )

        return output, final_hidden
