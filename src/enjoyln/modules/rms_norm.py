import torch
import torch.nn as nn
from torch.nn import functional as F

# import rms_norm_cpp
import rms_norm_cuda


def native_rms_norm_forward(
    self: nn.LayerNorm,
    x: torch.Tensor
) -> torch.Tensor:
    return torch.rms_norm(
        x, self.normalized_shape, self.weight, self.eps
    ) + self.bias


def rms_norm_forward(
    self: nn.LayerNorm,
    x: torch.Tensor
) -> torch.Tensor:
    output = rms_norm_cuda.forward(
        x, self.normalized_shape, self.weight, self.bias, self.eps
    )
    return output[0]


def replace_layer_norm_forward(
    layer: nn.LayerNorm,
    forward_fn: callable = rms_norm_forward,
    class_name: str = 'RMSNorm'
) -> None:
    layer.__class__ = type(
        class_name,
        (nn.Module,),
        {'forward': forward_fn}
    )
    return None


if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity, schedule
    from torch._tensor_str import printoptions
    torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=768)
    args = parser.parse_args()

    def check_diff(diff):
        mean_diff = torch.mean(diff.abs(), dim=-1, keepdim=True)
        max_diff, max_index = torch.max(diff.abs().flatten(), 0)
        with printoptions(precision=4, sci_mode=True):
            print("\n".join([
                f"Mean diff: {mean_diff.mean()}",
                f"Max diff: {max_diff}",
            ]))

    my_schedule = schedule(
        wait=1000,
        warmup=500,
        active=2500,
    )

    class MyModel(nn.Module):
        def __init__(self) -> None:
            super(MyModel, self).__init__()
            self.embed_dim = args.embed_dim
            self.eps = 1e-5
            self.norm = torch.nn.LayerNorm(self.embed_dim, eps=self.eps)
            self.norm.weight.data = torch.randn(self.embed_dim)
            self.norm.bias.data = torch.randn(self.embed_dim)

        def forward(self, x):
            return self.norm(x)

    model = MyModel().to('cuda')
    x = torch.randn(args.embed_dim).to('cuda')
    x -= x.mean()

    replace_layer_norm_forward(model.norm, forward_fn=native_rms_norm_forward)

    with profile(
        activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA
        ],
        schedule=my_schedule,
        record_shapes=True
    ) as prof:
        with torch.no_grad():
            with record_function("LayerNorm"):
                for _ in range(12000):
                    model(x)
                    torch.cuda.synchronize()
                    prof.step()
                native_rms_out = model(x)
                print(native_rms_out)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    replace_layer_norm_forward(model.norm, forward_fn=rms_norm_forward)

    with profile(
        activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA
        ],
        schedule=my_schedule,
        record_shapes=True
    ) as prof:
        with torch.no_grad():
            with record_function("RMSNorm"):
                for _ in range(12000):
                    model(x)
                    torch.cuda.synchronize()
                    prof.step()
                rms_out = model(x)
                print(rms_out)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    check_diff(native_rms_out - rms_out)
