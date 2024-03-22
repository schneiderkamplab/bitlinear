import torch
import torch.nn as nn
import torch.nn.functional as F

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, eps=1e-8, activation_bits=8, allow_zero=False):
        super(BitLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.eps = eps
        self.Q_b = 2**(activation_bits-1)
        self.allow_zero = allow_zero
        self.ones = torch.ones_like(self.weight)
        self.minus_ones = -1*self.ones

    def binarize_weights(self):
        if self.allow_zero:
            gamma = self.weight.abs().mean()
            unclipped = self.weight/(gamma+self.eps)
            binarized = torch.max(self.minus_ones, torch.min(self.ones, torch.round(unclipped)))
        else:
            alpha = self.weight.mean()
            binarized = torch.sign(self.weight-alpha)
        return binarized

    def quantize_activations(self, x):
        gamma = x.abs().max()
        quantized_x = torch.clamp(
                x*self.Q_b/(gamma+self.eps),
                -self.Q_b+self.eps,
                self.Q_b-self.eps,
            )
        return quantized_x

    def scale_activations(self, x):
        eta = x.min()
        gamma = x.abs().max()
        scaled_x = torch.clamp(
            (x-eta)*self.Q_b/(gamma+self.eps),
            self.eps,
            self.Q_b-self.eps,
        )
        return scaled_x

    def forward(self, input):
        binarized_weights = self.binarize_weights()
        output = F.linear(input, binarized_weights, self.bias)
        output = self.quantize_activations(output)
        #output = self.scale_activations(output)
        return output
