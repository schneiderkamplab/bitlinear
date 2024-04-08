import torch
import torch.nn as nn

from bitlinear import BitLinear, replace_layers, requantize_layers

#ACTIVATION_CLASS = nn.Sigmoid
ACTIVATION_CLASS = nn.ReLU
LAYER_CLASS = BitLinear
LAYER_KWARGS = {"allow_zero": True, "training": True, "auto_requantize": True}
#LAYER_CLASS = nn.Linear
#DATASET = 'mstz/breast'
DATASET = 'imodels/credit-card'
#TARGET_COL= 'is_cancer'
TARGET_COL= 'default.payment.next.month'

model, X_test, y_test = torch.load("model.pt")
print(model)
for _ in range(10):
    replace_layers(model, BitLinear, LAYER_CLASS, **LAYER_KWARGS)

    criterion = nn.HuberLoss()

    with torch.no_grad():
        output_test = model(X_test)
        acc_test = sum(1 for z, y in zip(output_test, y_test) if abs(z.item() - y.item()) < 0.5) / len(y_test)
        loss_test = criterion(output_test, y_test)
    print(f"test_loss {loss_test} test_acc {acc_test}")
