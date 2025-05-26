from .fp32 import FloatTransformer
from .quant import QuantTransformer


def build_model(model_params: dict):

    enable_qat = model_params["enable_qat"]
    if enable_qat:
        return QuantTransformer(**model_params)
    else:
        return FloatTransformer(**model_params)
