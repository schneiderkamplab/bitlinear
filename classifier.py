import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim, layer_class=nn.Linear, layer_kwargs={}, activation_class=nn.Sigmoid):
        super(Classifier, self).__init__()
        self.input = layer_class(input_dim, hidden_dim, **layer_kwargs)
        self.hidden_layers = nn.ModuleList([layer_class(hidden_dim, hidden_dim, **layer_kwargs) for _ in range(hidden_layers)])
        self.head = layer_class(hidden_dim, output_dim, **layer_kwargs)
        self.activation = activation_class()

    def forward(self, x):
        x = self.input(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.head(x)
        x = self.activation(x)
        return x
