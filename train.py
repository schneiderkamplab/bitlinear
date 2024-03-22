from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from bitlinear import BitLinear
from classifier import Classifier

EPOCHS = 100000
PATIENCE = 100
HIDDEN_DIM = 128
HIDDEN_LAYERS = 4
INTERVAL = 1
LEARNING_RATE = 1e-2
LR_SCHEDULER = False
#ACTIVATION_CLASS = nn.Sigmoid
ACTIVATION_CLASS = nn.ReLU
LAYER_CLASS = BitLinear
LAYER_KWARGS = {"allow_zero": True}
#LAYER_CLASS = nn.Linear
#DATASET = 'mstz/breast'
DATASET = 'imodels/credit-card'
#TARGET_COL= 'is_cancer'
TARGET_COL= 'default.payment.next.month'

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

model = Classifier(
    input_dim=X_train.shape[1],
    hidden_dim=HIDDEN_DIM,
    hidden_layers=HIDDEN_LAYERS,
    output_dim=1,
    layer_class=LAYER_CLASS,
    layer_kwargs=LAYER_KWARGS,
    activation_class=ACTIVATION_CLASS,
)

criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=0, patience=50)

best = 0
losses = []
acces = []

for epoch in tqdm(range(EPOCHS)):
    optimizer.zero_grad()
    output_train = model(X_train)
    acc_train = sum(1 for z, y in zip(output_train, y_train) if abs(z.item() - y.item()) < 0.5) / len(y_train)
    loss_train = criterion(output_train, y_train)
    loss_train.backward()
    optimizer.step()
    with torch.no_grad():
        output_test = model(X_test)
        acc_test = sum(1 for z, y in zip(output_test, y_test) if abs(z.item() - y.item()) < 0.5) / len(y_test)
        loss_test = criterion(output_test, y_test)
        if LR_SCHEDULER:
            scheduler.step(loss_test)
        losses.append(loss_test)
        acces.append(acc_test)
        if losses[best] > loss_test or acc_test == 1.0:
            best = epoch
        if epoch - best > PATIENCE or acc_test == 1.0:
            print(f"early stopping at epoch {epoch} with patience {PATIENCE}")
            break
    if epoch % INTERVAL == 0:
        print(f"epoch {epoch} train_loss {loss_train} test_loss {loss_test} train_acc {acc_train} test_acc {acc_test} lr {optimizer.param_groups[0]['lr']}")
print(f"best epoch {best} with loss {losses[best]} acc {acces[best]}")