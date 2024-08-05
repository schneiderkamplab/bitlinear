import torch

####### HELPERS #########
    
AbsMax = lambda input : input.abs().max()
AbsMedian = lambda input : input.abs().median()
AbsMean = lambda input : input.abs().mean()

def round_clamp(input, measure, eps) -> tuple[torch.Tensor, float]:
    scale = eval(measure)(input.detach()).clamp_(min=eps).item()
    scaled_input = input/scale
    return (scaled_input.round().clamp(-1, 1) - scaled_input).detach() + scaled_input, scale
    
