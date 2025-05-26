import torch
import torch.nn as nn

from .PE import PositionalEncoding
from .Encoder import Encoder


class FloatTransformer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        num_in_features = kwargs.get("num_in_features")
        d_model = kwargs.get("d_model")
        self.window_size = kwargs.get("window_size")
        num_out_features = kwargs.get("num_out_features")

        self.input_linear = nn.Linear(num_in_features, d_model)
        self.position_encoding = PositionalEncoding(
            d_model=d_model, max_window_size=self.window_size
        )
        self.encoder = Encoder(
            window_size=self.window_size,
            d_model=d_model,
            nhead=kwargs.get("nhead"),
            num_layers=kwargs.get("num_enc_layers"),
        )
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.output_linear = nn.Linear(d_model, num_out_features)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        inputs = self.input_linear(inputs)
        pos_info = self.position_encoding(self.window_size)
        inputs_pos = inputs + pos_info
        outputs = self.encoder(inputs_pos)
        outputs = self.pooling_layer(outputs.permute(0, 2, 1)).squeeze(2)
        outputs = self.output_linear(outputs)
        return outputs
