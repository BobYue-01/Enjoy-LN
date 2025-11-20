import torch
from transformers.pytorch_utils import Conv1D


def center_conv1d(layer: Conv1D) -> None:
    layer.weight.data -= layer.weight.data.mean(dim=-1, keepdim=True)
    if layer.bias is not None:
        layer.bias.data -= layer.bias.data.mean(dim=0, keepdim=True)
    return None


if __name__ == '__main__':
    conv = Conv1D(3, 5)
    print(conv.weight)
    print(conv.bias)
    center_conv1d(conv)
    print(conv.weight)
    print(conv.bias)
    x = torch.randn(4, 5)
    print(x)
    print(conv(x))
    print(conv(x).mean(dim=-1))
