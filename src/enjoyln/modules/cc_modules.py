import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from .cc_conv1d import center_conv1d
from .cc_embedding import center_embedding
from .cc_linear import center_linear


def center_modules(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        center_linear(layer)
    elif isinstance(layer, Conv1D):
        center_conv1d(layer)
    elif isinstance(layer, nn.Embedding):
        center_embedding(layer)
    return None
