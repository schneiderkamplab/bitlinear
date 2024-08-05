import torch
from math import ceil


'''
Activation Quantization Helpers
'''

def symmetric_range_from_bits(self, range):
    return (ceil(-2**(range-1)), ceil(2**(range-1)-1))

def round_clamp(self, input):
    return (input.round().clamp(self.range[0], self.range[1]) - input).detach() + input
    
class ActivationMeasure:
    def __init__(self, range=8, eps=1e-5):
        self.range = symmetric_range_from_bits(range)
        self.eps = eps
    
    def __call__(self, input):
        x_norm = torch.layer_norm(input, input.size()[1:])
        x_scale = self.scale(x_norm)
        return round_clamp(x_norm/x_scale), x_scale

    def scale(self, input) -> torch.Tensor:
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    

'''
Callable Activation Quantization Classes
'''

class Fp16(ActivationMeasure):
    def __init__(self, range=8, eps=1e-5):
        pass
    def __call__(self, input) -> tuple[torch.Tensor, torch.Tensor]:
        return input, torch.Tensor(1.0)

class AbsMax(ActivationMeasure): 
    def scale(self, input) -> torch.Tensor:
        return input.abs().max(dim=-1, keepdim=True).values.clamp_(min=self.eps)/max(abs(k) for k in self.range)

class AbsMean(ActivationMeasure):
    def scale(self, input) -> torch.Tensor:
        return input.abs().mean(dim=-1, keepdim=True).clamp_(min=self.eps)/max(abs(k) for k in self.range)

class AbsMedian(ActivationMeasure):
    def scale(self, input) -> torch.Tensor:
        return input.abs().median(dim=-1, keepdim=True).values.clamp_(min=self.eps)/max(abs(k) for k in self.range)


'''
Weight Quantization Functions
'''

class WeightMeasure:
    AbsMax = lambda input : input.abs().max()
    AbsMedian = lambda input : input.abs().median()
    AbsMean = lambda input : input.abs().mean()

def weight_round_clamp(input, measure, eps) -> tuple[torch.Tensor, float]:
    scale = eval(f'WeightMeasure.{measure}')(input.detach()).clamp_(min=eps)
    scaled_input = input/scale
    return (scaled_input.round().clamp(-1, 1) - scaled_input).detach() + scaled_input, scale