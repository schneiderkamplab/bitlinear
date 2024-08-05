from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bitlinear import BitLinear, replace_modules
from classifier import Classifier

# data config
DATASET = 'imodels/credit-card'
TARGET_COL= 'default.payment.next.month'

# training config
EPOCHS = 100000
BATCH_SIZE = 0 # 0 for no batching
PATIENCE = 100
INTERVAL = 100
LEARNING_RATE = 1e-2
SAVE = True
LR_SCHEDULER = False
try:
    import schedulefree
    OPTIMIZER_CLASS = schedulefree.AdamWScheduleFree
except ImportError:
    OPTIMIZER_CLASS = torch.optim.Adam

# model config
HIDDEN_DIM = 128
HIDDEN_LAYERS = 4
ACTIVATION_CLASS = nn.ReLU
LAYER_CLASS = BitLinear
LAYER_KWARGS = {"activation_measure": "AbsMax", "weight_measure": None, "weight_range": (-1, 1), "activation_range": [-7, 7]}

# data preparation
data = load_dataset(DATASET)['train'].to_pandas()
X = data.drop([TARGET_COL], axis=1).values
y = data[TARGET_COL].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train).unsqueeze(1)
y_test = torch.Tensor(y_test).unsqueeze(1)

# model preparation

model = Classifier(
    input_dim=X_train.shape[1],
    hidden_dim=HIDDEN_DIM,
    hidden_layers=HIDDEN_LAYERS,
    output_dim=1,
    layer_class=nn.Linear,
    layer_kwargs={"bias": False},
    activation_class=ACTIVATION_CLASS,
)
replace_modules(model, nn.Linear, LAYER_CLASS, LAYER_KWARGS)

# training preparation
criterion = nn.HuberLoss()
optimizer = OPTIMIZER_CLASS(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=0, patience=50)
best = 0
losses = []
acces = []
if BATCH_SIZE:
    xy_train = DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True)

# training
for epoch in tqdm(range(EPOCHS)):
    optimizer.zero_grad()
    model.train()
    if optimizer.__class__.__name__ == "AdamWScheduleFree":
        optimizer.train()
    if BATCH_SIZE:
        output_train = []
        for X_train_batch, y_train_batch in xy_train:
            output_train_batch = model(X_train_batch)
            loss_train = criterion(output_train_batch, y_train_batch)
            loss_train.backward()
            output_train.append(output_train_batch)
        output_train = torch.cat(output_train)
    else:
        output_train = model(X_train)
        loss_train = criterion(output_train, y_train)
        loss_train.backward()
    acc_train = sum(1 for z, y in zip(output_train, y_train) if abs(z.item() - y.item()) < 0.5) / len(y_train)
    optimizer.step()
    with torch.no_grad():
        model.eval()
        if optimizer.__class__.__name__ == "AdamWScheduleFree":
            optimizer.eval()
        output_test = model(X_test)
        acc_test = sum(1 for z, y in zip(output_test, y_test) if abs(z.item() - y.item()) < 0.5) / len(y_test)
        loss_test = criterion(output_test, y_test)
        if LR_SCHEDULER:
            scheduler.step(loss_test)
        losses.append(loss_test)
        acces.append(acc_test)
        if losses[best] > loss_test or acc_test == 1.0:
            best = epoch
            if SAVE:
                os.makedirs("../models", exist_ok=True)
                torch.save((model, X_test, y_test), "../models/classifier.pt")
                print(f"epoch {epoch} saved model with test_loss {loss_test} test_acc {acc_test}")
        if epoch - best > PATIENCE or acc_test == 1.0:
            print(f"early stopping at epoch {epoch} with patience {PATIENCE}")
            break
    if epoch % INTERVAL == 0:
        print(f"epoch {epoch} train_loss {loss_train} test_loss {loss_test} train_acc {acc_train} test_acc {acc_test} lr {optimizer.param_groups[0]['lr']}")
print(f"best epoch {best} with loss {losses[best]} acc {acces[best]}")
