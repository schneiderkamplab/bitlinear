import torch
from src.utils import Packed8, warptiling

class Warptiling(Packed8):
        
    def __call__(self, activations, weights, bias, scale):
            # Check constraints.
            assert activations.is_contiguous(), "Matrix A must be contiguous"
            assert activations.shape[0] == bias.shape[0], "Bias dimension must match input"
            
            M, K = activations.shape
            N = weights.shape[0] * 4//K    
            
            x_quant, x_scale = self.activations(activations)
            output = torch.zeros((M, N), device='cuda', dtype = torch.float16)
            
            # print(output, x_quant, weights, bias)
            
            warptiling.matmul(M, N, K, x_quant, weights, output)
            
            return (output + bias) * scale * x_scale