import torch
import torch.nn.functional as F

from src.utils import Default

class TorchLinear(Default):
    def __call__(self, input, weight, bias=None, scale=1):
        x_quant, x_scale = self.activations(input)
        return F.linear(x_quant, weight, bias) * scale * x_scale
