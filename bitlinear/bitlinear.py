from collections.abc import Sequence
from math import ceil
import re
import torch
import torch.nn as nn

from .kernels import TorchLinear
from .measures import AbsMax, AbsMedian, AbsMean

def round_clamp(input, range, lambda_=1):
    return lambda_ * (input.round().clamp(range[0], range[1]) - input).detach() + input

def scale(input, range, measure, keepdim, eps):
    return max(abs(k) for k in range) / measure(input.detach(), keepdim=keepdim).clamp_(min=eps)

def range_from_bits(bits):
    return (ceil(-2**(bits-1)), ceil(2**(bits-1)-1))

def sample(input, range, lambda_=1):
    if range[0] != -1 or range[1] != 1:
        return round_clamp(input, range, lambda_)
    rand = torch.rand(input.size(), device=input.device, dtype=input.dtype)
    result = input.sign() * torch.where(input.abs() < rand, 0, 1)
    return lambda_ * (result-input).detach() + input

class BitLinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            eps=1e-5,
            weight_range=1.58,
            weight_measure="AbsMedian",
            activation_range=8,
            activation_measure="AbsMax",
            kernel="TorchLinear",
            strategy="round_clamp",
            lambda_=1,
        ):
        super(BitLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.eps = eps
        self.weight_range = weight_range if isinstance(weight_range, Sequence) else range_from_bits(weight_range)
        self.weight_measure = eval(weight_measure)() if isinstance(weight_measure, str) else weight_measure
        self.activation_range = activation_range if isinstance(activation_range, Sequence) else range_from_bits(activation_range)
        self.activation_measure = eval(activation_measure)() if isinstance(activation_measure, str) else activation_measure
        self.kernel = eval(kernel)() if isinstance(kernel, str) else kernel
        self.strategy = eval(strategy) if isinstance(strategy, str) else strategy
        self.lambda_ = lambda_

    def __repr__(self):
        return f"BitLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, eps={self.eps}, weight_range={self.weight_range}, weight_measure={self.weight_measure}, activation_range={self.activation_range}, activation_measure={self.activation_measure}, kernel={self.kernel}, strategy={self.strategy})"

    def forward(self, x):
        if self.activation_measure is None:
            x_scale, x_quant = 1, x
        else:
            x_norm = torch.layer_norm(x, x.size()[1:])
            x_scale = scale(x_norm, self.activation_range, self.activation_measure, True, self.eps)
            x_quant = self.strategy(x_norm * x_scale, self.activation_range, self.lambda_)
        if self.weight_measure is None:
            w_scale, w_quant = 1, self.weight
        else:
            w_scale = scale(self.weight, self.weight_range, self.weight_measure, False, self.eps)
            w_quant = self.strategy(self.weight * w_scale, self.weight_range, self.lambda_)
        y_quant = self.kernel(x_quant, w_quant, self.bias)
        y = y_quant / (w_scale * x_scale)
        return y

class FrozenBitLinear(nn.Linear):
    def __init__(self, bitlinear):
        bias = bitlinear.bias is not None
        super(FrozenBitLinear, self).__init__(
            in_features=bitlinear.in_features,
            out_features=bitlinear.out_features,
            bias=bias,
            device=bitlinear.device,
            dtype=bitlinear.dtype,
        )
        if bias:
            self.bias.data = bitlinear.bias.data
        self.eps = bitlinear.eps
        self.activation_range = bitlinear.activation_range
        self.activation_measure = bitlinear.activation_measure
        if self.weight_measure is None:
            self.w_scale, self.w_quant = 1, bitlinear.weight
        else:
            self.w_scale = scale(bitlinear.weight, bitlinear.weight_range, bitlinear.weight_measure, False, self.eps)
            self.w_quant = self.strategy(bitlinear.weight * self.w_scale, bitlinear.weight_range)

    def forward(self, x):
        if self.activation_measure is None:
            x_scale, x_quant = 1, x
        else:
            x_norm = torch.layer_norm(x, x.size()[1:])
            x_scale = scale(x_norm, self.activation_range, self.activation_measure, True, self.eps)
            x_quant = self.strategy(x_norm * x_scale, self.activation_range)
        y_quant = self.kernel(x_quant, self.w_quant, self.bias)
        y = y_quant / (self.w_scale * x_scale)
        return y

def replace_modules(model, old_class=nn.Linear, new_class=BitLinear, new_class_kwargs={}, match_name="", prefix=""):
    for name, module in model.named_children():
        qual_name = prefix + "." + name
        if isinstance(module, old_class) and re.search(match_name, qual_name) is not None:
            kwargs = dict(new_class_kwargs)
            kwargs["in_features"] = module.in_features
            kwargs["out_features"] = module.out_features
            bias = getattr(module, "bias", None) is not None
            kwargs["bias"] = bias
            kwargs["device"] = module.weight.device
            new_module = new_class(**kwargs)
            new_module.weight.data = module.weight.data
            if bias:
                new_module.bias.data = module.bias.data
            setattr(model, name, new_module)
        else:
            replace_modules(module, old_class, new_class, new_class_kwargs, match_name, prefix=qual_name)

def bitlinearize(model, old_class=nn.Linear, new_class=BitLinear, replacements=[{}]):
    for replacement in replacements:
        replacement = dict(replacement)
        match_name = replacement.pop("match_name", "")
        replace_modules(
            model=model,
            old_class=old_class,
            new_class=new_class,
            new_class_kwargs=replacement,
            match_name=match_name,
        )
    return model

def set_lamnda_(model, lambda_=1):
    assert 0.0 <= lambda_ <= 1.0
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.lambda_ = lambda_
