import torch
import torch.nn.functional as F

from src.utils import Default

class TorchLinear(Default):
    
    def __call__(self, input, weight, bias=None, scale=1):
        return F.linear(input, weight, bias) * scale
    

class TorchMulAdd(TorchLinear):
    def __call__(self, input, weight, bias=None, scale=1):
        output = input @ weight.t()
        if bias is not None:
            output += bias
        return output

class Naive(TorchLinear):
    def __call__(self, input, weight, bias=None, scale=1):
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
        return torch.Tensor(output) * scale

class NaiveListComp(TorchLinear):
    def __call__(self, input, weight, bias=None, scale=1):
        input = input.tolist()
        weight = weight.tolist()
        if bias is not None:
            bias = bias.tolist()
        output = []
        n = len(input)
        m = len(input[0])
        p = len(weight)
        output = [[sum(input[i][k] * weight[j][k] for k in range(m)) + (bias[j] if bias is not None else 0) for j in range(p)] for i in range(n)]
        return torch.Tensor(output) * scale

class TernaryNaive(TorchLinear):
    def __call__(self, input, weight, bias=None, scale=1):
        input = input.tolist()
        weight = weight.tolist()
        assert all(all(x in {-1, 0, 1} for x in row) for row in input)
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
                    match weight[j][k]:
                        case 1:
                            value += input[i][k]
                        case -1:
                            value -= input[i][k]
                if bias is not None:
                    value += bias[j]
                out.append(value)
            output.append(out)
        return torch.Tensor(output) * scale
