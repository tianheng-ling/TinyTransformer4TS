import torch
import torch.nn as nn

from .MHA import MultiHeadAttention
from .FFN import FeedForwardNetwork


class EncoderLayer(nn.Module):
    def __init__(
        self,
        window_size: int,
        d_model: int,
        nhead: int,
    ) -> None:
        super().__init__()

        self.window_size = window_size

        self.mha = MultiHeadAttention(d_model, nhead)
        self.mha_norm = nn.BatchNorm1d(d_model)
        self.ffn = FeedForwardNetwork(d_model)
        self.ffn_norm = nn.BatchNorm1d(d_model)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:

        mha_outputs, attn = self.mha(inputs, inputs, inputs)
        mha_add_outputs = inputs + mha_outputs
        mha_norm_outputs = self.mha_norm(mha_add_outputs.permute(0, 2, 1)).permute(
            0, 2, 1
        )

        ffn_outputs = self.ffn(mha_norm_outputs)
        ffn_add_outputs = mha_norm_outputs + ffn_outputs
        ffn_norm_outputs = self.ffn_norm(ffn_add_outputs.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        return ffn_norm_outputs, attn
