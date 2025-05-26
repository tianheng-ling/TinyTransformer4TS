import torch
import torch.nn as nn
import math

from config import DEVICE


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_window_size: int):

        super().__init__()
        position = torch.arange(0, max_window_size).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        self.register_buffer(
            "pe",
            torch.zeros(
                (max_window_size, d_model),
                dtype=torch.float32,
            ).to(DEVICE),
        )
        self.pe.requires_grad = False
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, len_input: torch.FloatTensor) -> torch.FloatTensor:
        return self.pe[:, :len_input]
