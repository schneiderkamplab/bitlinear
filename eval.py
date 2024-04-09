import torch
import torch.nn as nn

from bitlinear2 import BitLinear, replace_layers
#from bitnet import BitLinear
#from bitnet_int import BitLinear
import kernels

LAYER_CLASS = BitLinear
LAYER_KWARGS = {
#    "allow_zero": True,
#    "training": False,
#    "auto_requantize": False,
#    "kernel": kernels.torch_linear,
#    "kernel": kernels.torch_mul_add,
#    "kernel": kernels.naive,
#    "kernel": kernels.naive_listcomp,
    "kernel": kernels.ternary_naive,
}

model, X_test, y_test = torch.load("model.pt")
print(model)
replace_layers(model, BitLinear, LAYER_CLASS, **LAYER_KWARGS)
print(model)

criterion = nn.HuberLoss()

with torch.no_grad():
    output_test = model(X_test)
    acc_test = sum(1 for z, y in zip(output_test, y_test) if abs(z.item() - y.item()) < 0.5) / len(y_test)
    loss_test = criterion(output_test, y_test)
print(f"test_loss {loss_test} test_acc {acc_test}")
