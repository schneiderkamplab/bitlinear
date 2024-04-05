import torch
import torch.nn as nn
import torch.nn.functional as F

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        alpha = input.mean()
        binarized = torch.sign(input-alpha)
        binarized.requires_grad = False
        return binarized, alpha

    @staticmethod
    def backward(ctx, grad_output, alpha):
        return grad_output, None

class Ternarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ones, minus_ones, eps=1e-5):
        gamma = input.abs().mean()
        unclipped = input/(gamma+eps)
        ternarized = torch.max(minus_ones, torch.min(ones, torch.round(unclipped)))
        ternarized.requires_grad = False
        return ternarized, gamma

    @staticmethod
    def backward(ctx, grad_output, gamma):
        return grad_output, None, None, None

class AbsMaxQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, eps=1e-5, activation_bits=8):
        Q_b = 2**(activation_bits-1)
        gamma = input.abs().max()
        quantized = torch.round(
            torch.clamp(
                input*Q_b/gamma,
                -Q_b+eps,
                Q_b-eps,
            ),
        )
        quantized.requires_grad = False
        return quantized, gamma

    @staticmethod
    def backward(ctx, grad_output, gamma):
        return grad_output, None, None

class MinScaleQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, eps=1e-5, activation_bits=8):
        Q_b = 2**(activation_bits-1)
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
        scaled.requires_grad = False
        return scaled, gamma

    @staticmethod
    def backward(ctx, grad_output, gamma):
        return grad_output, None, None

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
        self.ones = torch.ones_like(self.weight)
        self.minus_ones = -1*self.ones
        self.requantize()

    def requantize(self):
        self.quantized_weights, self.beta = Ternarize.apply(self.weight, self.ones, self.minus_ones, self.eps) if self.allow_zero else Binarize.apply(self.weight, self.eps)

    def forward(self, input):
        normalized_activations = torch.layer_norm(input, input.size()[1:])
        quantized_activations, gamma = AbsMaxQuantize.apply(normalized_activations, self.eps, self.activation_bits)
        self.requantize()
        quantized_outputs = F.linear(quantized_activations, self.quantized_weights, self.bias)
        dequantized_output = quantized_outputs*self.beta*gamma/2**(self.activation_bits-1)
        return dequantized_output

def replace_layers(model, old_class, new_class, **new_class_kwargs):
    for name, module in model.named_children():
        if isinstance(module, old_class):
            kwargs = dict(new_class_kwargs)
            kwargs["in_features"] = module.in_features
            kwargs["out_features"] = module.out_features
            kwargs["bias"] = module.bias is not None
            setattr(model, name, new_class(**kwargs))
            print(f"replaced layer {name} of class {old_class} with {new_class}")
        else:
            replace_layers(module, old_class, new_class, **new_class_kwargs)

def requantize_layers(model):
    for name, module in model.named_children():
        if isinstance(module, BitLinear):
            module.requantize()
            #print(f"requantized layer {name}")
        else:
            requantize_layers(module)
