from torch import nn

def center_embedding(layer: nn.Embedding) -> None:
    layer.weight.data -= layer.weight.data.mean(dim=-1, keepdim=True)
    return None


if __name__ == '__main__':
    emb = nn.Embedding(10, 3)
    print(emb.weight)
    center_embedding(emb)
    print(emb.weight)
    print(emb.weight.mean(dim=-1))
