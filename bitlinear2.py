import torch
import torch.nn as nn

from kernels import torch_linear

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
            activation_bits=8,
            auto_requantize=True,
            integer=True,
            training=True,
            kernel=torch_linear,
        ):
        super(BitLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.eps = eps
        self.activation_bits = activation_bits
        self.auto_requantize = auto_requantize
        self.integer = integer
        self.training = training
        self.kernel = kernel
        self.Q_b = 2**(activation_bits-1)-1
        if not self.auto_requantize or not self.training:
            self.requantize()

    def __repr__(self):
        return f"BitLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, eps={self.eps}, activation_bits={self.activation_bits}, auto_requantize={self.auto_requantize}, training={self.training}, kernel={self.kernel})"

    def requantize(self):
        self.w_scale = 1 / self.weight.abs().mean().clamp_(min=self.eps)
        self.w_quant = round_clamp(self.weight * self.w_scale, -1, 1)
        if not self.integer:
            self.w_quant = self.w_quant / self.w_scale

    def forward(self, x):
        x_norm = torch.layer_norm(x, x.size()[1:])
        x_scale = self.Q_b / x_norm.abs().max(dim=-1, keepdim=True).values.clamp_(min=self.eps)
        x_quant = round_clamp(x_norm * x_scale, -self.Q_b-1, self.Q_b)
        if not self.integer:
            x_quant = x_quant / x_scale
        if self.training and self.auto_requantize:
            self.requantize()
        y_quant = self.kernel(x_quant, self.w_quant, self.bias)
        y = y_quant / (self.w_scale * x_scale) if self.integer else y_quant
        return y

def replace_layers(model, old_class, new_class, **new_class_kwargs):
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
            if hasattr(module, "norm"):
                new_module.norm = module.norm
            setattr(model, name, new_module)
            #print(f"replaced layer {name} of class {old_class} with {new_class}")
        else:
            replace_layers(module, old_class, new_class, **new_class_kwargs)

def requantize_layers(model):
    for name, module in model.named_children():
        if isinstance(module, BitLinear):
            if not module.auto_requantize:
                module.requantize()
        else:
            requantize_layers(module)
