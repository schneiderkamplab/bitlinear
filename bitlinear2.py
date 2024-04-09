import torch
import torch.nn as nn

from kernels import torch_linear

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
        gamma = input.abs().max(dim=-1, keepdim=True).values
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
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            eps=1e-5,
            activation_bits=8,
            allow_zero=True,
            auto_requantize=True,
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
        self.allow_zero = allow_zero
        self.auto_requantize = auto_requantize
        self.training = training
        self.kernel = kernel
        self.ones = torch.ones_like(self.weight)
        self.minus_ones = -1*self.ones
        if not self.auto_requantize or not self.training:
            self.requantize()

    def __repr__(self):
        return f"BitLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, eps={self.eps}, activation_bits={self.activation_bits}, allow_zero={self.allow_zero}, auto_requantize={self.auto_requantize}, training={self.training}, kernel={self.kernel})"

    def requantize(self):
        if self.ones.device != self.weight.data.device or self.ones.device != self.weight.data.device:
            self.ones = self.ones.to(self.weight.data.device)
            self.minus_ones = self.minus_ones.to(self.weight.data.device)
        self.quantized_weights, self.beta = Ternarize.apply(self.weight, self.ones, self.minus_ones, self.eps) if self.allow_zero else Binarize.apply(self.weight, self.eps)

    def forward(self, input):
        normalized_activations = torch.layer_norm(input, input.size()[1:])
        quantized_activations, gamma = AbsMaxQuantize.apply(normalized_activations, self.eps, self.activation_bits)
        if self.training and self.auto_requantize:
            self.requantize()
        quantized_outputs = self.kernel(quantized_activations, self.quantized_weights, self.bias)
        dequantized_output = quantized_outputs*self.beta*gamma/2**(self.activation_bits-1)
        return dequantized_output

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
