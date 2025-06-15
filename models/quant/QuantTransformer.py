import torch
import torch.nn as nn

from models.quant.design import Transformer as TransformerDesign
from elasticai.creator.nn.integer.linear import Linear
from models.fp32.PE import PositionalEncoding
from elasticai.creator.nn.integer.addition import Addition
from elasticai.creator.nn.integer.encoder import Encoder
from elasticai.creator.nn.integer.avgpooling1d import AVGPooling1d
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class QuantTransformer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        num_in_features = kwargs.get("num_in_features")
        window_size = kwargs.get("window_size")
        d_model = kwargs.get("d_model")
        ffn_dim = kwargs.get("ffn_dim", 4 * d_model)
        num_out_features = kwargs.get("num_out_features")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.do_int_forward = kwargs.get("do_int_forward")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        # prepare positional encoding info
        pos_encoding = PositionalEncoding(d_model=d_model, max_window_size=window_size)
        self.inputs_pos_info = pos_encoding(len_input=window_size)

        # prepare layers
        self.input_linear = Linear(
            name="input_linear",
            in_features=num_in_features,
            out_features=d_model,
            num_dimensions=window_size,
            bias=True,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.pos_info_addition = Addition(
            name="add_pos_info",
            num_features=d_model,
            num_dimensions=window_size,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.encoder = Encoder(
            name="encoder",
            window_size=window_size,
            d_model=d_model,
            ffn_dim=ffn_dim,
            nhead=kwargs.get("nhead"),
            num_enc_layers=kwargs.get("num_enc_layers"),
            quant_bits=self.quant_bits,
            do_int_forward=self.do_int_forward,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.avg_pooling = AVGPooling1d(  # assume (batch_size, channels, window_size)
            name="average_pooling",
            in_features=d_model,
            out_features=d_model,
            in_num_dimensions=window_size,
            out_num_dimensions=1,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.output_linear = Linear(
            name="output_linear",
            in_features=d_model,
            out_features=num_out_features,
            num_dimensions=1,
            bias=True,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.precomputed = False

    def create_design(self, name: str) -> TransformerDesign:
        return TransformerDesign(
            name=name,
            data_width=self.quant_bits,
            input_linear=self.input_linear,
            inputs_pos_info=self.q_inputs_pos_info.squeeze(0).tolist(),
            pos_info_addition=self.pos_info_addition,
            encoder=self.encoder,
            avg_pooling=self.avg_pooling,
            output_linear=self.output_linear,
            work_library_name="work",
        )

    def precompute(self) -> None:

        self.input_linear.precompute()
        self.q_inputs_pos_info = self.pos_info_addition.inputs2_QParams.quantize(
            self.inputs_pos_info
        )
        self.pos_info_addition.precompute()
        self.encoder.sequential.precompute()
        self.avg_pooling.precompute()
        self.output_linear.precompute()

        self.precomputed = True

    def forward(
        self,
        inputs: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # inputs shape: (batch_size, window_size, num_in_features)
        if self.do_int_forward:
            assert not self.training, "int_forward() can only be used in inference mode"
            self.precompute()

            # Quantize inputs
            q_inputs = self.input_linear.inputs_QParams.quantize(inputs).to("cpu")
            save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

            q_inputs = self.input_linear.int_forward(q_inputs=q_inputs)
            q_pos_inputs = self.pos_info_addition.int_forward(
                q_inputs1=q_inputs, q_inputs2=self.q_inputs_pos_info
            )
            q_encoder_outputs = self.encoder.int_forward(
                q_inputs=q_pos_inputs,
            )
            q_avg_pooling_outputs = self.avg_pooling.int_forward(
                q_inputs=q_encoder_outputs
            )
            q_outputs = self.output_linear.int_forward(
                q_inputs=q_avg_pooling_outputs,
            )
            save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
            # Dequantize outputs
            dq_outputs = self.output_linear.outputs_QParams.dequantize(q_outputs)
            return dq_outputs

        else:
            inputs = self.input_linear.forward(inputs, given_inputs_QParams=None)
            pos_inputs = self.pos_info_addition.forward(
                inputs1=inputs,
                inputs2=self.inputs_pos_info,
                given_inputs1_QParams=self.input_linear.outputs_QParams,
                given_inputs2_QParams=None,
            )
            encoder_outputs = self.encoder.forward(
                inputs=pos_inputs,
                given_inputs_QParams=self.pos_info_addition.outputs_QParams,
            )
            avg_pooling_outputs = self.avg_pooling.forward(
                inputs=encoder_outputs,
                given_inputs_QParams=self.encoder.outputs_QParams,
            )
            outputs = self.output_linear.forward(
                inputs=avg_pooling_outputs,
                given_inputs_QParams=self.avg_pooling.outputs_QParams,
            )
            return outputs
