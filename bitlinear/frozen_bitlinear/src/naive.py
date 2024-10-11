from src.utils import Packed8, naive

class Naive(Packed8):
        fxn = naive.linear
        
        def __call__(self, activations, weights, bias, scale):
                # Check constraints.
                assert activations.is_contiguous(), "Matrix A must be contiguous"
                assert activations.shape[0] == bias.shape[0], "Bias dimension must match input"
                
                M, K = activations.shape
                N = weights.shape[0] * 4//K    
                
                x_quant, x_scale = self.activations(activations)
                
                return self.fxn(x_quant, weights, bias, M, N, K) * scale * x_scale










