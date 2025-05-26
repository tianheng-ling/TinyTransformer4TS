import torch
import torch.nn as nn

from .EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        window_size: int,
        d_model: int,
        nhead: int,
    ) -> None:
        super().__init__()

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    window_size,
                    d_model,
                    nhead,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        attn_weights = []
        for layer in self.encoder_layers:
            output, attn = layer(input)
            attn_weights.append(attn)
        return output
