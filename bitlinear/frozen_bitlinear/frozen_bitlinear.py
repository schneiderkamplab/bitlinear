import torch
import torch.nn as nn

from ..bitlinear import BitLinear
from ..measures import *
from bitlinear.frozen_bitlinear.frozen_bitlinear import TorchLinear, Naive

class FrozenBitLinear(nn.Linear):
    
    w_scale = None
    
    def __init__(
            self,
            in_features,
            out_features,
            kernel,
            bias=True,
            eps=1e-5,
            activation_range=8,
            activation_measure='AbsMax',
            device=None,
            dtype=None
        ):
        super(FrozenBitLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        
        self.eps = eps
        
        self.kernel = eval(kernel)(activation_range, activation_measure) if isinstance(kernel, str) else kernel        

    def __repr__(self):
        return f"FrozenBitLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, kernel={self.kernel}), activation_range={self.activation_range}, activation_measure={self.activation_measure}"

    def forward(self, x):
        if self.activation_measure is None:
            x_scale, x_quant = 1, x
        else:
            x_norm = torch.layer_norm(x, x.size()[1:])
            x_scale = 1 / scale(x_norm, self.activation_range, self.activation_measure, True, self.eps)
            x_quant = round_clamp(x_norm / x_scale, self.activation_range)
        
        return self.kernel(x_quant, self.weight, self.bias, self.w_scale * x_scale)
    
    def freeze_weights(self, weights:torch.Tensor, weightMeasure:str='AbsMean'):
        """
        Parameters:
            weights : torch.Tensor
                tensor of weights to be packed (out_features x in_features)
            weightMeasure : str
                str corresponding to a weight packing method
                options are 'AbsMean', 'AbsMax', and 'AbsMedian'
                default is 'AbsMean'

        Returns:
            None
        
        This function must be called for a Frozen BitLinear Module to pack the weights. It takes advantage of a 
        kernel-specific weight packing function to store the weights in their ternary representation. This is implicitly 
        called in bitlinear.freeze
        """
        
        assert (weights.shape[0] == self.out_features) & (weights.shape[1] == self.in_features), "Weights dimensions must match that of the layer"

        packed_weights, self.w_scale = self.kernel.scale_weights(weights, weightMeasure, self.eps)
        
        if isinstance(packed_weights, torch.nn.Parameter):
            self.weight = packed_weights
        else:
            self.weight = nn.Parameter(packed_weights)

    
def freeze(
    model, 
    kernel=TorchLinear, 
    weightMeasure = 'AbsMean', 
    eps=1e-5, 
    activation_measure='AbsMax', 
    activation_range=16, 
    device=None, 
    dtype=None
    ):
    """
        Parameters:
            model : torch.nn.Module
                Model with BitLinear modules to replace with FrozenBitLinear for faster inference
            kernel : str or Kernel
                Forward kernel to implement in the model
            weightMeasure : str
                str corresponding to a weight packing method
                options are 'AbsMean', 'AbsMax', and 'AbsMedian'
                default is 'AbsMean'
            eps : float
                value to clamp the weights to in the scaling step to avoid divide-by-zero
                default is 1e-5
            activation_measure : str
                str corresponding to the activation quantization method
                options are 'AbsMean', 'AbsMax', 'AbsMedian', and None
                'AbsMax'
            activation_range : int
                number of bits to represent the activations in
                default is 8
            device : Optional(torch.device)
            dtype : Optional(torch.dtype)

        Returns:
            None
        
        This function replaces all of the BitLinear instances in a model with FrozenBitLinear in order to
        speed up inference. The weights are packed corresponding to the kernel.
        """
    
    for name, module in model.named_children():
        if isinstance(module, BitLinear):
            
            kwargs = {
                'kernel' : kernel,
                'in_features' : module.in_features,
                'out_features' : module.out_features,
                'bias' : getattr(module, "bias", None) is not None,
                'device' : device,
                'dtype' : dtype,
                'eps' : eps,
                'activation_measure' : activation_measure,
                'activation_range' : activation_range
            }
            
            frozen_module = FrozenBitLinear(**kwargs)
            frozen_module.freeze_weights(module.weight.data, weightMeasure) # pack 
            
            if kwargs['bias']:
                frozen_module.bias.data = module.bias.data
            
            setattr(model, name, frozen_module)
            
        else: # recursively iterate throughout the rest of the model
            freeze(module, kernel, weightMeasure, eps, device, dtype)
