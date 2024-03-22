import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim, layer_class=nn.Linear):
        super(Classifier, self).__init__()
        self.input = layer_class(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([layer_class(hidden_dim, hidden_dim) for _ in range(hidden_layers)])
        self.head = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.head(x)
        x = self.activation(x)
        return x
