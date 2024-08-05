import torch
import os
import json

def weights(N, K, kernel, baseline, dtype = torch.float16) -> tuple[torch.Tensor, torch.Tensor, float]:
    path = f'weights/{kernel}/{N}_{K}'
    try:
        
        torch_weights = torch.load(os.path.join(path, 'torch.pt'), weights_only=True).to('cuda')
        kernel_weights = torch.load(os.path.join(path, 'kernel.pt'), weights_only=True).to('cuda')
        with open(os.path.join(path, 'scale.json'), 'r') as f:
            scale = json.load(f)
        return torch_weights, kernel_weights, scale
    
    except Exception as e:
        print(f"Could not load weights because of {e}, generating new ones at {path}")
        
        weights = torch.randn((N, K), device='cuda', dtype=dtype)
        (base_weights, scale) =  baseline.scale_weights(weights)
        (kernel_weights, _) = kernel.scale_weights(weights)
        
        os.makedirs(path, exist_ok=True)
        torch.save(base_weights, os.path.join(path, 'torch.pt'))
        torch.save(kernel_weights, os.path.join(path, 'kernel.pt'))
        with open(os.path.join(path, 'scale.json'), 'w') as f:
            json.dump(scale, f)
        
        return base_weights, kernel_weights, scale
    
