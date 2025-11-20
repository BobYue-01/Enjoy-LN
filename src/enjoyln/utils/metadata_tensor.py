import torch


class MetadataTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data: torch.Tensor, centered=False, last_modules=set()):
        requires_grad = data.requires_grad
        self = torch.Tensor._make_subclass(
            cls, data, require_grad=requires_grad
        )
        return self

    def __init__(self, data: torch.Tensor, centered=False, last_modules=set()):
        self.centered = centered
        self.last_modules = last_modules

    def __getattr__(self, name):
        if name == 'centered':
            return False
        elif name == 'last_modules':
            return set()
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    def __add__(self, other):
        result = super(MetadataTensor, self).__add__(other)
        if isinstance(other, MetadataTensor):
            centered = self.centered and other.centered
            last_modules = self.last_modules | other.last_modules
        else:
            centered = False
            last_modules = self.last_modules
        return MetadataTensor(
            result, centered=centered, last_modules=last_modules)

    def __repr__(self):
        if hasattr(self, "centered"):
            centered = self.centered
        else:
            centered = None
        return f"MetadataTensor({super().__repr__()}, " + \
            f"centered={centered}, last_modules={self.last_modules})"


if __name__ == "__main__":
    x1 = MetadataTensor(
        torch.tensor([1, 2, 3]),
        centered=True,
        last_modules={"module1", "module2"}
    )
    x2 = MetadataTensor(
        torch.tensor([4, 5, 6]),
        centered=True,
        last_modules={"module3"}
    )
    y1 = x1 + x2
    print(y1)

    x3 = torch.tensor([7, 8, 9])
    y2 = y1 + x3
    print(y2)
