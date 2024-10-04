import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SGConv

import torch.nn.functional as F

from bitlinear import BitLinear, replace_modules


USE_BITLINEAR = False   # Toggle this
BITLINEAR_WEIGHT_MEASURE = "AbsMean"  # AbsMean, AbsMedian
K = 2
lr = 0.01

DATASET = "PubMed" # Cora, Citeseer, PubMed
dataset = Planetoid(root=f"/tmp/Planetoid/{DATASET}", name=DATASET)  

class BitSGConv(SGConv):
    def __init__(self, in_channels: int, out_channels: int, K=1, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(in_channels, out_channels, K=K, bias=bias, **kwargs)
        self.lin = BitLinear(in_channels, out_channels, bias=bias, weight_measure=BITLINEAR_WEIGHT_MEASURE, strategy="sample")
        self.reset_parameters()


# Initialize the model and optimizer
sgc_kwargs = {'K': K, 'add_self_loops': True, 'cached': True, 'aggr': 'sum'}
if USE_BITLINEAR:
    model = BitSGConv(dataset.num_features, dataset.num_classes, **sgc_kwargs)
else:
    model = SGConv(dataset.num_features, dataset.num_classes, **sgc_kwargs)
    
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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
for epoch in range(100):
    train_loss = train()
    acc = test()
    print(f'Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Test Accuracy: {acc:.4f}')