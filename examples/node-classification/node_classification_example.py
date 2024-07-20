import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SGConv

import torch.nn.functional as F

from bitlinear import BitLinear, replace_modules


USE_BITLINEAR = True   # Toggle this

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')

class BitSGConv(SGConv):
    def __init__(self, in_channels: int, out_channels: int, K=1, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(in_channels, out_channels, K=K, bias=bias, **kwargs)
        self.lin = BitLinear(in_channels, out_channels, bias=bias)
        self.reset_parameters()


# Initialize the model and optimizer
if USE_BITLINEAR:
    model = BitSGConv(dataset.num_features, dataset.num_classes, K=1, add_self_loops=True)
else:
    model = SGConv(dataset.num_features, dataset.num_classes, K=1, add_self_loops=True)
    
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(model)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(dataset[0].x, dataset[0].edge_index)
    loss = criterion(out[dataset[0].train_mask], dataset[0].y[dataset[0].train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation loop
def test():
    model.eval()
    out = model(dataset[0].x, dataset[0].edge_index)
    pred = out.argmax(dim=1)
    acc = pred[dataset[0].test_mask] == dataset[0].y[dataset[0].test_mask]
    acc = int(acc.sum()) / int(dataset[0].test_mask.sum())
    return acc

# Train and evaluate the model
for epoch in range(400):
    train_loss = train()
    acc = test()
    print(f'Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Test Accuracy: {acc:.4f}')