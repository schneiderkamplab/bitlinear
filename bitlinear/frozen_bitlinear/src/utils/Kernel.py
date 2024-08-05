import torch
import torch.nn.functional as F

from src.utils.helpers import weight_round_clamp, AbsMax, AbsMean, AbsMedian, Fp16
from cuda.pack_weights import pack_weights
   
class Kernel:
    def __init__(self, eps=1e-5, activation_range = 8, activation_measure = 'AbsMax'):
        
        self.eps = eps
        self.activations = eval(activation_measure)(activation_range, eps)
        
    def __call__(self, activations, weights, bias, scale) -> torch.Tensor:
        raise NotImplementedError # this needs to be changed depending on the kernel
    
    def scale_weights(self, weights, measure='AbsMean') -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError # this needs to be changed depending on the kernel
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    
class Default(Kernel):
    def scale_weights(self, weights, measure='AbsMean') -> tuple[torch.Tensor, torch.Tensor]:
        return weight_round_clamp(weights, measure, self.eps)

class Packed8(Kernel):
    def scale_weights(self, weights, measure='AbsMean') -> tuple[torch.Tensor, torch.Tensor]:
        weights, scale = weight_round_clamp(weights, measure, self.eps)
        return pack_weights.packedint8(weights, *weights.shape), scale
