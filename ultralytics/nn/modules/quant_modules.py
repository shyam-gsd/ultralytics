import copy
import math
from abc import abstractmethod, ABCMeta
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import UpsamplingNearest2d
from torch.nn.functional import interpolate

import brevitas.nn as qnn
import  brevitas.quant as quant
from brevitas import config
from brevitas.core.function_wrapper import CeilSte
from brevitas.inject.enum import RestrictValueType
import  torch.nn as nn



__all__ = (
    "QC3",
    "QC3k",
    "QC2f",
    "QC3k2",
    "QuantSPPF",
    "QuantAttention",
    "QPSABlock",
    "QuantBottleneck",
    "QC2PSA",
    "QDWConv",
    "QuantDFL",
    "QuantDetect",
    "QuantConv",
    "Uint8ActPerTensorPoT",
    "Int8ActPerTensorPoT",
    "Int8WeightPerChannelPoT",
    "QuantUpsamplingNearest2d"

)

from brevitas.nn.mixin import QuantLayerMixin
from brevitas.quant_tensor import QuantTensor
from ultralytics.utils.tal import make_anchors, dist2bbox


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
    default_act = qnn.QuantSigmoid(act_quant=Uint8ActPerTensorPoT,bit_width= 6,return_quant_tensor=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True ,**kwargs):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.bit_width = kwargs.get('bit_width', 6)
        self.act_quant = kwargs.get("act_quant",Int8ActPerTensorPoT)

        self.weight_quant = kwargs.get("weight_quant",Int8WeightPerChannelPoT)
        self.weight_bit_width = kwargs.get('weight_bit_width', 6)
        self.return_quant_tensor = kwargs.get('return_quant_tensor', True)

        self.conv = qnn.QuantConv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False, weight_quant=self.weight_quant,weight_bit_width=self.weight_bit_width,return_quant_tensor=self.return_quant_tensor,input_quant=Int8ActPerTensorPoT,output_quant=Int8ActPerTensorPoT)
        self.bn = nn.BatchNorm2d(c2)
        default_act = qnn.QuantSigmoid(act_quant=self.act_quant,bit_width=self.bit_width,return_quant_tensor=self.return_quant_tensor)  # default activation
        self.act = default_act if act is True else act if isinstance(act, nn.Module) else qnn.QuantIdentity(return_quant_tensor=self.return_quant_tensor,act_quant=self.act_quant,bit_width=self.bit_width)


    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""

        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class QuantBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5,**kwargs):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = QuantConv(c1, c_, k[0], 1,**kwargs)
        self.cv2 = QuantConv(c_, c2, k[1], 1, g=g,**kwargs)
        self.add = shortcut and c1 == c2
        self.requantize = qnn.QuantIdentity(return_quant_tensor=True,act_quant=Int8ActPerTensorPoT,bit_width= 6)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return self.requantize(x) + self.requantize(self.cv2(self.cv1(x))) if self.add else self.requantize(self.cv2(self.cv1(x)))

class QC3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,**kwargs):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = QuantConv(c1, c_, 1, 1,**kwargs)
        self.cv2 = QuantConv(c1, c_, 1, 1,**kwargs)
        self.cv3 = QuantConv(2 * c_, c2, 1,**kwargs)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(QuantBottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0,**kwargs) for _ in range(n)))
        self.requantize = qnn.QuantIdentity(return_quant_tensor=True, act_quant=Int8ActPerTensorPoT, bit_width=6)
    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.requantize(self.m(self.cv1(x))), self.requantize(self.cv2(x))), 1))

class QC3k(QC3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3,**kwargs):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e,**kwargs)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(QuantBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0,**kwargs) for _ in range(n)))

class QC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,**kwargs):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = QuantConv(c1, 2 * self.c, 1, 1,**kwargs)
        self.cv2 = QuantConv((2 + n) * self.c, c2, 1,**kwargs)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(QuantBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0,**kwargs) for _ in range(n))
        self.requantize = qnn.QuantIdentity(return_quant_tensor=True, act_quant=Int8ActPerTensorPoT, bit_width=6)

    def forward(self, x):
        """Forward pass through C2f layer."""
        chunks = self.cv1(x).chunk(2, 1)
        y = list(self.requantize(chunks[i]) for i in range(2))
        y.extend(self.requantize(m(y[-1])) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        chunks = self.cv1(x).split((self.c, self.c), 1)
        y = list(chunks[i] for i in range(2))
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class QC3k2(QC2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True,**kwargs):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e,**kwargs)
        self.m = nn.ModuleList(
            QC3k(self.c, self.c, 2, shortcut, g,**kwargs) if c3k else QuantBottleneck(self.c, self.c, shortcut, g,**kwargs) for _ in range(n)
        )

class QuantSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5,**kwargs):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = QuantConv(c1, c_, 1, 1,**kwargs)
        self.cv2 = QuantConv(c_ * 4, c2, 1, 1,**kwargs)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class QuantAttention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5,**kwargs):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = QuantConv(dim, h, 1, act=False,**kwargs)
        self.proj = QuantConv(dim, dim, 1, act=False,**kwargs)
        self.pe = QuantConv(dim, dim, 3, 1, g=dim, act=False,**kwargs)
        self.requantize = qnn.QuantIdentity(return_quant_tensor=True, act_quant=Int8ActPerTensorPoT, bit_width=6)
        self.dequantize = qnn.QuantIdentity(act_quant=None)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.dequantize(self.qkv(x))
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = self.requantize((v @ attn.transpose(-2, -1)).view(B, C, H, W)) + self.requantize(self.pe(v.reshape(B, C, H, W)))
        x = self.proj(x)
        return x

class QPSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True,**kwargs) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = QuantAttention(c, attn_ratio=attn_ratio, num_heads=num_heads,**kwargs)
        self.ffn = nn.Sequential(QuantConv(c, c * 2, 1), QuantConv(c * 2, c, 1, act=False,**kwargs))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class QC2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5,**kwargs):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = QuantConv(c1, 2 * self.c, 1, 1,**kwargs)
        self.cv2 = QuantConv(2 * self.c, c1, 1,**kwargs)

        self.m = nn.Sequential(*(QPSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64,**kwargs) for _ in range(n)))
        self.requantize = qnn.QuantIdentity(return_quant_tensor=True, act_quant=Int8ActPerTensorPoT, bit_width=6)
        self.dequantize = qnn.QuantIdentity(act_quant=None)
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.dequantize(self.cv1(x)).split((self.c, self.c), dim=1)
        b = self.requantize(self.m(b))
        return self.cv2(torch.cat((self.requantize(a), b), 1))

class QDWConv(QuantConv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True,**kwargs):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act,**kwargs)



class QuantDFL(nn.Module):

    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16,**kwargs):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.bit_width = kwargs.get('bit_width', 6)
        if "weight_quant" in kwargs:
            self.weight_quant = kwargs['weight_quant']
        else:
            self.weight_quant = None

        self.weight_bit_width = kwargs.get('weight_bit_width', 6)
        self.return_quant_tensor = kwargs.get('return_quant_tensor', True)
        self.conv = qnn.QuantConv2d(c1, 1, 1, bias=False,weight_quant=self.weight_quant,input_quant=Int8ActPerTensorPoT,output_quant=Int8ActPerTensorPoT,weight_bit_width=self.weight_bit_width,return_quant_tensor=self.return_quant_tensor).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        #x = torch.quantize_per_tensor(c1,scale=0,1, zero_point=0, dtype=torch.quint8)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class QuantDetect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=(),**kwargs):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels

        self.weight_quant = kwargs.get("weight_quant",Int8WeightPerChannelPoT)
        self.weight_bit_width = kwargs.get('weight_bit_width', 6)
        self.return_quant_tensor = kwargs.get('return_quant_tensor', True)
        self.act_quant = kwargs.get("act_quant",Int8ActPerTensorPoT)



        self.cv2 = nn.ModuleList(
            nn.Sequential(QuantConv(x, c2, 3,**kwargs),
                          QuantConv(c2, c2, 3,**kwargs),
                          qnn.QuantConv2d(c2, 4 * self.reg_max, 1,
                                          weight_quant=self.weight_quant,
                                          weight_bit_width=self.weight_bit_width,
                                          return_quant_tensor=self.return_quant_tensor,
                                          input_quant=self.act_quant,
                                          output_quant=self.act_quant)
                          ) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(QuantConv(x, c3, 3,**kwargs),
                                        QuantConv(c3, c3, 3,**kwargs),
                                        qnn.QuantConv2d(c3, self.nc, 1,
                                                        weight_quant=self.weight_quant,
                                                        weight_bit_width=self.weight_bit_width,
                                                        return_quant_tensor=self.return_quant_tensor,
                                                        input_quant=self.act_quant,
                                                        output_quant=self.act_quant)
                                        ) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(QDWConv(x, x, 3,**kwargs), QuantConv(x, c3, 1,**kwargs)),
                    nn.Sequential(QDWConv(c3, c3, 3,**kwargs), QuantConv(c3, c3, 1,**kwargs)),
                    qnn.QuantConv2d(c3, self.nc, 1,
                                    weight_quant=self.weight_quant,
                                    weight_bit_width=self.weight_bit_width,
                                    return_quant_tensor=self.return_quant_tensor,
                                    input_quant=self.act_quant,
                                    output_quant=self.act_quant),
                )
                for x in ch
            )
        )
        self.dfl = QuantDFL(self.reg_max,**kwargs) if self.reg_max > 1 else qnn.QuantIdentity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

        self.requantize = qnn.QuantIdentity(return_quant_tensor=True, act_quant=Int8ActPerTensorPoT, bit_width=6)
        self.dequantize = qnn.QuantIdentity(act_quant=None)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = self.dequantize(torch.cat((self.requantize(self.cv2[i](x[i])), self.requantize(self.cv3[i](x[i]))), 1))
        if self.training or isinstance(x[0], torch.fx.Proxy):  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)

class ExportMixin(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._export_mode = False
        self.export_debug_name = None
        self.export_handler = None
        self.export_input_debug = False
        self.export_output_debug = False

    @property
    @abstractmethod
    def requires_export_handler(self):
        pass

    @property
    def export_mode(self):
        return self._export_mode

    @export_mode.setter
    def export_mode(self, value):
        if value and config.JIT_ENABLED:
            raise RuntimeError(
                "Export mode with BREVITAS_JIT is currently not supported. Save the model' "
                "state_dict to a .pth, load it back with BREVITAS_JIT=0, and call export.")
        if value and self.training:
            raise RuntimeError("Can't enter export mode during training, only during inference")
        if value and self.requires_export_handler and self.export_handler is None:
            raise RuntimeError("Can't enable export mode on a layer without an export handler")
        elif value and not self.requires_export_handler and self.export_handler is None:
            return  # don't set export mode when it's not required and there is no handler
        elif value and not self._export_mode and self.export_handler is not None:
            self.export_handler.prepare_for_export(self)
            self.export_handler.attach_debug_info(self)
        elif not value and self.export_handler is not None:
            self.export_handler = None
        self._export_mode = value


class _CachedIO:

    def __init__(self, quant_tensor: QuantTensor, metadata_only: bool):
        self.shape = quant_tensor.value.shape
        if metadata_only:
            self.quant_tensor = quant_tensor.set(value=None)
        else:
            self.quant_tensor = quant_tensor

    @property
    def scale(self):
        return self.quant_tensor.scale

    @property
    def zero_point(self):
        return self.quant_tensor.zero_point

    @property
    def bit_width(self):
        return self.quant_tensor.bit_width

    @property
    def signed(self):
        return self.quant_tensor.signed


class QuantLayerMixin(ExportMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            return_quant_tensor: bool,
            cache_inference_quant_inp: bool = False,
            cache_inference_quant_out: bool = False,
            cache_quant_io_metadata_only: bool = True):
        ExportMixin.__init__(self)
        self.accept_quant_tensor = True
        self.return_quant_tensor = return_quant_tensor
        self.cache_inference_quant_inp = cache_inference_quant_inp
        self.cache_inference_quant_out = cache_inference_quant_out
        self.cache_quant_io_metadata_only = cache_quant_io_metadata_only
        self._cached_inp = None
        self._cached_out = None

    @property
    @abstractmethod
    def channelwise_separable(self) -> bool:
        pass

    @property
    def is_quant_input_signed(self) -> Optional[bool]:  # tri-valued logic output
        if self._cached_inp is not None:
            return self._cached_inp.signed
        else:
            return None

    def _set_global_is_quant_layer(self, value):
        config._IS_INSIDE_QUANT_LAYER = value

    def quant_input_scale(self):
        if self._cached_inp is not None:
            return self._cached_inp.scale
        else:
            return None

    def quant_input_zero_point(self):
        if self._cached_inp is not None:
            return self._cached_inp.zero_point
        else:
            return None

    def quant_input_bit_width(self):
        if self._cached_inp is not None:
            return self._cached_inp.bit_width
        else:
            return None

    @property
    def is_quant_output_signed(self) -> Optional[bool]:  # tri-valued logic output
        if self._cached_out is not None:
            return self._cached_out.signed
        else:
            return None

    def quant_output_scale(self):
        if self._cached_out is not None:
            return self._cached_out.scale
        else:
            return None

    def quant_output_zero_point(self):
        if self._cached_out is not None:
            return self._cached_out.zero_point
        else:
            return None

    def quant_output_bit_width(self):
        if self._cached_out is not None:
            return self._cached_out.bit_width
        else:
            return None

    def unpack_input(self, inp: Union[Tensor, QuantTensor]):
        self._set_global_is_quant_layer(True)
        # Hack to recognize a QuantTensor that has decayed to a tuple
        # when used as input to tracing (e.g. during ONNX export)
        if (torch._C._get_tracing_state() is not None and isinstance(inp, tuple) and
                len(inp) == len(QuantTensor._fields) and all([isinstance(t, Tensor) for t in inp])):
            inp = QuantTensor(*inp)
        if isinstance(inp, QuantTensor):
            # don't cache values during export pass
            if not self.training and not self._export_mode and self.cache_inference_quant_inp:
                cached_inp = _CachedIO(inp.detach(), self.cache_quant_io_metadata_only)
                self._cached_inp = cached_inp
        else:
            inp = QuantTensor(inp, training=self.training)
            if not self.training and self.cache_inference_quant_inp:
                cached_inp = _CachedIO(inp.detach(), self.cache_quant_io_metadata_only)
                self._cached_inp = cached_inp
        # Remove any naming metadata to avoid dowmstream errors
        # Avoid inplace operations on the input in case of forward hooks
        if not torch._C._get_tracing_state():
            inp = inp.set(value=inp.value.rename(None))
        return inp

    def pack_output(self, quant_output: QuantTensor):
        if not self.training and self.cache_inference_quant_out:
            self._cached_out = _CachedIO(quant_output.detach(), self.cache_quant_io_metadata_only)
        self._set_global_is_quant_layer(False)
        if self.return_quant_tensor:
            return quant_output
        else:
            return quant_output.value


class QuantUpsamplingNearest2d(QuantLayerMixin, UpsamplingNearest2d):

    def __init__(self, size=None, scale_factor=None, return_quant_tensor: bool = True, **kwargs):
        UpsamplingNearest2d.__init__(self, size=size, scale_factor=scale_factor)
        QuantLayerMixin.__init__(self, return_quant_tensor)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def toggle_quantize(self, quantize):
        pass

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(x.value)
            self._set_global_is_quant_layer(False)
            return out
        y_value = interpolate(x.value, self.size, self.scale_factor, self.mode, self.align_corners)
        y = x.set(value=y_value)
        return self.pack_output(y)