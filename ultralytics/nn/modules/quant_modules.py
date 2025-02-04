import brevitas.nn as qnn
import  brevitas.quant as quant
from brevitas.core.function_wrapper import CeilSte
from brevitas.inject.enum import RestrictValueType
import  torch.nn as nn



__all__ = (

    "QuantConv",
    "Uint8ActPerTensorPoT",
    "Int8ActPerTensorPoT",
    "Int8WeightPerChannelPoT",

)



class Uint8ActPerTensorPoT(quant.Uint8ActPerTensorFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = CeilSte

class Int8ActPerTensorPoT(quant.Int8ActPerTensorFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = CeilSte

class Int8WeightPerChannelPoT(quant.Int8WeightPerChannelFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = CeilSte

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class QuantConv(nn.Module):

    """Simplified RepConv module with Conv fusing."""
    default_act = qnn.QuantReLU(act_quant=Uint8ActPerTensorPoT,bit_width= 6,return_quant_tensor=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True ,weight_quant=None,act_quant=None,**kwargs):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.bit_width = kwargs.get('bit_width', 6)
        self.act_quant = act_quant

        self.weight_quant = weight_quant
        self.weight_bit_width = kwargs.get('weight_bit_width', 6)
        self.return_quant_tensor = kwargs.get('return_quant_tensor', True)

        self.conv = qnn.QuantConv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False, weight_quant=self.weight_quant,weight_bit_width=self.weight_bit_width,return_quant_tensor=self.return_quant_tensor,input_quant=Int8ActPerTensorPoT,output_quant=Int8ActPerTensorPoT)
        self.bn = nn.BatchNorm2d(c2)
        default_act = qnn.QuantReLU(act_quant=self.act_quant,bit_width=self.bit_width,return_quant_tensor=self.return_quant_tensor)  # default activation
        self.act = default_act if act is True else act if isinstance(act, nn.Module) else qnn.QuantIdentity(return_quant_tensor=self.return_quant_tensor,act_quant=self.act_quant,bit_width=self.bit_width)


    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""

        return self.act(self.bn(self.conv(x)))

    def toggle_quantize(self, quantize):
        if quantize:
            self.conv.weight_quant = self.weight_quant
            self.conv.weight_bit_width = self.weight_bit_width

            self.act.act_quant = self.act_quant
            self.act.bit_width = self.bit_width

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))