import torch
import torch.nn.functional as F

def torch_linear(input, weight, bias=None):
    return F.linear(input, weight, bias)

def torch_mul_add(input, weight, bias=None):
    output = input @ weight.t()
    if bias is not None:
        output += bias
    return output

first = True

def naive(input, weight, bias=None):
    print(input.shape, weight.shape, bias.shape if bias is not None else None)
    input = input.tolist()
    weight = weight.tolist()
    if bias is not None:
        bias = bias.tolist()
    output = []
    n = len(input)
    m = len(input[0])
    p = len(weight)
    for i in range(n):
        out = []
        for j in range(p):
            value = sum(input[i][k] * weight[j][k] for k in range(m))
            if bias is not None:
                value += bias[j]
            out.append(value)
        output.append(out)
    return torch.Tensor(output)

def naive_listcomp(input, weight, bias=None):
    print(input.shape, weight.shape, bias.shape if bias is not None else None)
    input = input.tolist()
    weight = weight.tolist()
    if bias is not None:
        bias = bias.tolist()
    output = []
    n = len(input)
    m = len(input[0])
    p = len(weight)
    output = [[sum(input[i][k] * weight[j][k] for k in range(m)) + (bias[j] if bias is not None else 0) for j in range(p)] for i in range(n)]
    return torch.Tensor(output)

def ternary_naive(input, weight, bias=None):
    count = 0
    total = 0
    print(input.shape, weight.shape, bias.shape if bias is not None else None)
    input = input.tolist()
    weight = weight.tolist()
    if bias is not None:
        bias = bias.tolist()
    output = []
    n = len(input)
    m = len(input[0])
    p = len(weight)
    for i in range(n):
        out = []
        for j in range(p):
            value = 0
            for k in range(m):
                total += 1
                match weight[j][k]:
                    case 1:
                        value += input[i][k]
                    case -1:
                        value -= input[i][k]
                    case 0:
                        count += 1
            if bias is not None:
                value += bias[j]
            out.append(value)
        output.append(out)
    print(count/total*100)
    return torch.Tensor(output)