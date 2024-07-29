import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from bitlinear import BitLinear, replace_modules, AbsMax, AbsMean, AbsMedian
import random
import numpy as np
import torch


### BEGIN from bitlinear source, could be imported
def round_clamp(input, range):
    return (input.round().clamp(range[0], range[1]) - input).detach() + input

def scale(input, range, measure, keepdim, eps):
    return max(abs(k) for k in range) / measure(input.detach(), keepdim=keepdim).clamp_(min=eps)
### END from bitlinear source


def quantize_weights(m: nn.Module):
    w_scale = scale(m.weight, m.weight_range, m.weight_measure, False, m.eps)
    w_quant = round_clamp(m.weight * w_scale, m.weight_range)
    return w_quant



def main():
    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n = 10000
    n_train = 5000
    n_test = n - n_train
    n_epochs = 1000
    bsz = 100
    break_on_first_solution = False

    lr = 0.01

    n_hidden = 8  # BitLinear seems to need more than Linear
    use_bitlinear = False  # change to false for basic MLP
    weight_measure = "AbsMedian"  # AbsMean, AbsMedian

    # Build toy dataset
    x = (torch.rand(n, 4) < 0.5)
    y = torch.logical_xor(x[:, 0], x[:, 1]).long()
    x = x.float()

    print("Y mean (should be about 0.5):", y.float().mean())

    # split..
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]

    if use_bitlinear:
        # Basic MLP
        model = nn.Sequential(BitLinear(4,n_hidden,weight_measure=weight_measure), nn.ReLU(), BitLinear(n_hidden, 2, weight_measure=weight_measure))
    else:
        # BitLinear MLP
        model = nn.Sequential(nn.Linear(4,n_hidden), nn.ReLU(), nn.Linear(n_hidden, 2))


    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=bsz)

    for i in range(n_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            print(loss.item())

        # eval
        model.eval()
        y_hat = model(x_test)
        acc = ((torch.argmax(y_hat, dim=-1) == y_test).sum().float() / y_test.size(0))
        print(f"Epoch {i} Accuracy: {acc.item()}")
        if break_on_first_solution and (acc - 1.0).abs() < 1e-8:
            break


    with torch.no_grad():
        # Check if inputs 2-3 are ignored
        print("Model", model, sep='\n')
        print("Model weights:")
        for i, m in enumerate(model.modules()):
            print(m)
            if not isinstance(m, BitLinear):
                print("Skipping.")
                continue
            # print("BitLinear shadow weights", m.weight, sep="\n")
            w_quant = quantize_weights(m)
            print("BitLinear quantized weights", w_quant, sep="\n")
            print("w_quant.size()", w_quant.size())

        print("-"*80)
        print("Analyzing input2hidden layer...", model[0])
        i2h_w = quantize_weights(model[0]) if isinstance(model[0], BitLinear) else model[0].weight
        print("Input2hidden weights cols 0-1 (should be xor)", i2h_w[:, [0,1]], sep="\n")
        print("Input2hidden weights cols 0-1 L1 norm:", torch.linalg.vector_norm(i2h_w[:, [0,1]], 1).item())
        print("-"*80)
        print("Input2hidden weights cols 2-3 (should be zero)", i2h_w[:, [2,3]], sep="\n")
        print("Input2hidden weights cols 2-3 L1 norm:", torch.linalg.vector_norm(i2h_w[:, [2,3]], 1).item())
        print("-"*80)
        print("Analyzing input2hidden bias...", model[0].bias)
        print("-"*80)

        print("-"*80)
        print("Analyzing hidden2output layer...", model[2])

        h2i_w = quantize_weights(model[-1]) if isinstance(model[-1], BitLinear) else model[-1].weight
        print("hidden2output weights", h2i_w, sep="\n")
        print("hidden2output weights L1 norm:", torch.linalg.vector_norm(h2i_w, 1).item())
        print("-"*80)
        print("Analyzing hidden2output bias...", model[2].bias)
        print("-"*80)

if __name__ == "__main__":
    main()
