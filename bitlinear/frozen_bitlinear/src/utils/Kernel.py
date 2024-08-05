import torch
import torch.nn.functional as F

from src.utils.helpers import round_clamp
from cuda.pack_weights import pack_weights
   
class Kernel:
        
    def __call__(self, activations, weights, bias, scale) -> torch.Tensor:
        raise NotImplementedError # this needs to be changed depending on the kernel
    
    def scale_weights(self, weights, measure='AbsMean', eps=1e-5) -> tuple[torch.Tensor, float]:
        raise NotImplementedError # this needs to be changed depending on the kernel
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    
class Default(Kernel):
    def scale_weights(self, weights, measure='AbsMean', eps=1e-5) -> tuple[torch.Tensor, float]:
        return round_clamp(weights, measure, eps)

class Packed8(Kernel):
    def scale_weights(self, weights, measure='AbsMean', eps=1e-5) -> tuple[torch.Tensor, float]:
        weights, scale = round_clamp(weights, measure, eps)
        return pack_weights.packedint8(weights, *weights.shape), scale
