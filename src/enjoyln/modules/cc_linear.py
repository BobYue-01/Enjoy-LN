import torch
from torch import nn


def center_linear(layer: nn.Linear) -> None:
    layer.weight.data -= layer.weight.data.mean(dim=0, keepdim=True)
    if layer.bias is not None:
        layer.bias.data -= layer.bias.data.mean(dim=0, keepdim=True)
    return None


if __name__ == '__main__':
    lin = nn.Linear(5, 3)
    print(lin.weight)
    print(lin.bias)
    center_linear(lin)
    print(lin.weight)
    print(lin.bias)
    x = torch.randn(4, 5)
    print(lin(x))
    print(lin(x).mean(dim=-1))
