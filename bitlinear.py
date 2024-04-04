import torch
import torch.nn as nn
import torch.nn.functional as F

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        alpha = input.mean()
        binarized = torch.sign(input-alpha)
        binarized.requires_grad = False
        return binarized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Ternarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, eps=1e-5):
        gamma = input.abs().mean()
        unclipped = input/(gamma+eps)
        ones = torch.ones_like(input)
        minus_ones = -1*ones
        ternarized = torch.max(minus_ones, torch.min(ones, torch.round(unclipped)))
        ternarized.requires_grad = False
        return ternarized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class AbsMaxQuantize(torch.autograd.Function):
    eps = 1e-5
    b = 8
    @staticmethod
    def forward(ctx, input):
        Q_b = 2**(AbsMaxQuantize.b-1)
        gamma = input.abs().max()
        quantized = torch.round(
            torch.clamp(
                input*Q_b/gamma,
                -Q_b+AbsMaxQuantize.eps,
                Q_b-AbsMaxQuantize.eps,
            ),
        )
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class MinScaleQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, eps=1e-5, b=8):
        Q_b = 2**(b-1)
        eta = input.min()
        positive = input-eta
        gamma = input.abs().max()
        scaled = torch.round(
            torch.clamp(
                positive*Q_b/gamma,
                eps,
                Q_b-eps,
            ),
        )
        return scaled

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, eps=1e-5, activation_bits=8, allow_zero=True):
        super(BitLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.eps = eps
        self.activation_bits = activation_bits
        self.allow_zero = allow_zero

    def forward(self, input):
        normalized_activations = torch.layer_norm(input, input.size()[1:])
        quantized_activations = AbsMaxQuantize.apply(normalized_activations)
        quantized_weights = Ternarize.apply(self.weight) if self.allow_zero else Binarize.apply(self.weight)
        quantized_outputs = F.linear(quantized_activations, quantized_weights, self.bias)
        gamma = normalized_activations.abs().mean()
        dequantized_output = quantized_outputs*self.weight.abs().mean()*gamma/2**(self.activation_bits-1)
        return dequantized_output

def replace_layer(model, old_class, new_class, **new_class_kwargs):
    for name, module in model.named_children():
        if isinstance(module, old_class):
            kwargs = dict(new_class_kwargs)
            kwargs["in_features"] = module.in_features
            kwargs["out_features"] = module.out_features
            kwargs["bias"] = module.bias is not None
            setattr(model, name, new_class(**kwargs))
            print(f"replaced layer {name} of class {old_class} with {new_class}")
        else:
            replace_layer(module, old_class, new_class, **new_class_kwargs)