import torch
from transformers.utils import ModelOutput

from .metadata_tensor import MetadataTensor


class Counter:
    def __init__(self):
        self.ln_cnt = 0
        self.foldable_cnt = 0
        self.center_modules = set()
        self.layernorms = set()
        self.layer_info = []
        self.indent = 0


class HookManager:
    def __init__(
        self,
        model: torch.nn.Module,
        hook_fn=None,
        hook_pre_fn=None,
        module_to_hook=None
    ) -> None:
        self.model = model
        self.hook_fn = hook_fn or (lambda module, inputs, outputs: outputs)
        self.hook_pre_fn = hook_pre_fn or (lambda module, inputs: None)
        self.module_to_hook = module_to_hook
        self.hooks = []

    def __enter__(self):
        if self.module_to_hook is None:
            for _, layer in self.model.named_modules():
                self.hooks.append(
                    layer.register_forward_pre_hook(self.hook_pre_fn))
                self.hooks.append(
                    layer.register_forward_hook(self.hook_fn))
        else:
            for layer in self.module_to_hook:
                self.hooks.append(
                    layer.register_forward_pre_hook(self.hook_pre_fn))
                self.hooks.append(
                    layer.register_forward_hook(self.hook_fn))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()


def create_analyse_hook_fns(counter: Counter, _print = True):
    def print_fn(*args, **kwargs):
        if _print:
            print(*args, **kwargs)
        else:
            pass

    def apply_func_to_nested_tuple(t, func):
        if isinstance(t, tuple):
            return tuple(apply_func_to_nested_tuple(x, func) for x in t)
        else:
            return func(t)

    def get_shape(t):
        if isinstance(t, torch.Tensor):
            return tuple(t.size())
        else:
            return None

    def get_centered(t):
        if isinstance(t, MetadataTensor) and hasattr(t, 'centered'):
            return t.centered
        else:
            return None

    def get_last_modules(t):
        if isinstance(t, MetadataTensor) and hasattr(t, 'last_modules'):
            return t.last_modules
        else:
            return set()

    def hook_pre_fn(module, inputs):
        print_fn('  ' * counter.indent, '< ', module.__class__.__name__, '>')
        counter.indent += 1
        for name, sub_module in module.named_children():
            print_fn('  ' * counter.indent, name, ':',
                  sub_module.__class__.__name__)

        inputs_centered = True
        last_modules = set()

        def input_func(input):
            nonlocal inputs_centered
            nonlocal last_modules
            if isinstance(input, MetadataTensor):
                inputs_centered = inputs_centered and input.centered
                last_modules |= input.last_modules
            else:
                inputs_centered = False
            print_fn('  ' * counter.indent, '<!-- <|', input.__class__.__name__, get_centered(input),
                  get_shape(input), len(get_last_modules(input)), get_last_modules(input))

        apply_func_to_nested_tuple(inputs, input_func)

        module._input_centered = inputs_centered
        module._last_modules = last_modules

        if 'LayerNorm' in module.__class__.__name__:
            counter.ln_cnt += 1
            if inputs_centered:
                counter.layernorms.add(module)
                counter.foldable_cnt += 1
                counter.center_modules |= last_modules

    def hook_fn(module, inputs, outputs):
        if isinstance(outputs, ModelOutput):
            return outputs

        single = False
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
            single = True

        def output_func(output):
            if not isinstance(output, MetadataTensor) and isinstance(output, torch.Tensor):
                output = MetadataTensor(output, centered=False)
            if isinstance(output, MetadataTensor):
                module_name = module.__class__.__name__
                if (
                    'LayerNorm' in module_name or
                    'Linear' in module_name or
                    'Conv' in module_name or
                    'Embed' in module_name
                ):
                    output.centered = True
                    output.last_modules = {module}
                elif (
                    'Drop' in module_name
                ):
                    output.centered = module._input_centered
                    output.last_modules = module._last_modules
                print_fn('  ' * counter.indent, '<!-- |>',  output.__class__.__name__, get_centered(output),
                      get_shape(output), len(get_last_modules(output)), get_last_modules(output))

            return output

        new_outputs = apply_func_to_nested_tuple(outputs, output_func)

        counter.indent -= 1
        print_fn('  ' * counter.indent, '</', module.__class__.__name__, '>')

        if single:
            new_outputs = new_outputs[0]
        else:
            new_outputs = tuple(new_outputs)

        return new_outputs

    return hook_pre_fn, hook_fn
