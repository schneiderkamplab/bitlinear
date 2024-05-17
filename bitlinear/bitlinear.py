from math import ceil
import torch
import torch.nn as nn

from .kernels import TorchLinear

def round_clamp(input, min, max):
    return (input.round().clamp(min, max) - input).detach() + input

class BitLinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            eps=1e-5,
            weight_bits=1.58,
            activation_bits=8,
            x_max=None,
            x_min=None,
            w_max=None,
            w_min=None,
            kernel=TorchLinear(),
            measure=torch.median,
        ):
        super(BitLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.eps = eps
        self.kernel = kernel
        self.measure = measure
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits
        self.x_max = x_max if x_max is not None else ceil(2**(activation_bits-1)-1)
        self.x_min = x_min if x_min is not None else ceil(-2**(activation_bits-1))
        self.w_max = w_max if w_max is not None else ceil(2**(weight_bits-1)-1)
        self.w_min = w_min if w_min is not None else ceil(-2**(weight_bits-1))

    def __repr__(self):
        return f"BitLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, eps={self.eps}, weight_bits={self.weight_bits}, activation_bits={self.activation_bits}, x_max={self.x_max}, x_min={self.x_min}, w_max={self.w_max}, w_min={self.w_min}, kernel={self.kernel}, measure={self.measure})"

    def forward(self, x):
        x_norm = torch.layer_norm(x, x.size()[1:])
        x_scale = self.x_max / x_norm.detach().abs().max(dim=-1, keepdim=True).values.clamp_(min=self.eps)
        x_quant = round_clamp(x_norm * x_scale, -self.x_min, self.x_max)
        w_scale = 1 / self.measure(self.weight.detach().abs()).clamp_(min=self.eps)
        w_quant = round_clamp(self.weight * w_scale, self.w_min, self.w_max)
        y_quant = self.kernel(x_quant, w_quant, self.bias)
        y = y_quant / (w_scale * x_scale)
        return y

def replace_modules(model, old_class=nn.Linear, new_class=BitLinear, new_class_kwargs={}):
    for name, module in model.named_children():
        if isinstance(module, old_class):
            kwargs = dict(new_class_kwargs)
            kwargs["in_features"] = module.in_features
            kwargs["out_features"] = module.out_features
            bias = getattr(module, "bias", None) is not None
            kwargs["bias"] = bias
            new_module = new_class(**kwargs)
            new_module.weight.data = module.weight.data
            if bias:
                new_module.bias.data = module.bias.data
            setattr(model, name, new_module)
        else:
            replace_modules(module, old_class, new_class, new_class_kwargs)
