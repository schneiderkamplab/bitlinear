from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from bitlinear import BitLinear, replace_layers, requantize_layers

#ACTIVATION_CLASS = nn.Sigmoid
ACTIVATION_CLASS = nn.ReLU
LAYER_CLASS = BitLinear
LAYER_KWARGS = {"allow_zero": True, "training": False}
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

model = torch.load("model.pt")
print(model)
replace_layers(model, BitLinear, LAYER_CLASS, **LAYER_KWARGS)
print(model)

criterion = nn.HuberLoss()

with torch.no_grad():
    output_train = model(X_train)
    acc_train = sum(1 for z, y in zip(output_train, y_train) if abs(z.item() - y.item()) < 0.5) / len(y_train)
    loss_train = criterion(output_train, y_train)
    output_test = model(X_test)
    acc_test = sum(1 for z, y in zip(output_test, y_test) if abs(z.item() - y.item()) < 0.5) / len(y_test)
    loss_test = criterion(output_test, y_test)
print(f"train_loss {loss_train} test_loss {loss_test} train_acc {acc_train} test_acc {acc_test}")
