import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from bitlinear import BitLinear, replace_modules

def main():
    n = 1000
    n_train = 500
    n_test = n - n_train
    n_epochs = 1000
    bsz = 100
    break_on_first_solution = False

    n_hidden = 16  # BitLinear seems to need more than Linear
    use_bitlinear = True  # change to false for basic MLP

    x = (torch.rand(n, 2) < 0.5)
    y = torch.logical_xor(x[:, 0], x[:, 1]).long()
    x = x.float()

    print("Y mean (should be about 0.5):", y.float().mean())

    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]

    if use_bitlinear:
        # Basic MLP
        model = nn.Sequential(BitLinear(2,n_hidden), nn.ReLU(), BitLinear(n_hidden, 2))
    else:
        # BitLinear MLP
        model = nn.Sequential(nn.Linear(2,n_hidden), nn.ReLU(), nn.Linear(n_hidden, 2))

    optimizer = AdamW(model.parameters(), lr=1e-3)
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
        y_hat = model(x_test)
        acc = ((torch.argmax(y_hat, dim=-1) == y_test).sum().float() / y_test.size(0))
        print(f"Epoch {i} Accuracy: {acc.item()}")
        if break_on_first_solution and (acc - 1.0).abs() < 1e-8:
            break


if __name__ == "__main__":
    main()
