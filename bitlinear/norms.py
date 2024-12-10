import torch

class Norm:
    def __init__(self, in_features):
        self.in_features = in_features
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class LayerNorm(Norm):
    def __call__(self, input):
        return torch.layer_norm(input, [self.in_features])

class ParametricLayerNorm(Norm, torch.nn.LayerNorm):
    def __init__(self, in_features):
        torch.nn.LayerNorm.__init__(self, in_features)
