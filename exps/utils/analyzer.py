import os
import json
import time

import torch
from torch.nn import Module
from torch.profiler import profile, ProfilerActivity, schedule

from enjoyln import utils
from enjoyln import modules
from utils.checker import CheckItem, Checker


class Analyzer:
    def __init__(
        self,
        model: Module,                # model to be folded and accelerated
        ref_model: Module = None,     # original model for comparison
        batch_size: int = 1,          # batch size for input
        seq_length: int = 512,        # sequence length for input
        abs_tol: float = 1e-5,        # absolute tolerance for equality checks
        replace: bool = False,        # replace outputs when they are close enough, to reduce error accumulation
        speed_iter: int = 16,         # number of iterations for speed tests
        print_output: bool = True,    # print output during analysis
    ):
        self.model = model
        self.ref_model = ref_model
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.abs_tol = abs_tol
        self.replace = replace
        self.speed_iter = speed_iter
        self.print_output = print_output

        self.output_queue = []

        self.model.eval()
        if self.ref_model is not None:
            self.ref_model.load_state_dict(self.model.state_dict())
            self.ref_model.eval()

        input_ids = torch.zeros(1, 1, dtype=torch.int, requires_grad=False).cuda()
        self.input_ids = input_ids
        self.meta_input_ids = utils.MetadataTensor(input_ids, centered=False).cuda()

        self.folded_counter = utils.Counter()
        self.folded_hook_pre_fn, self.folded_hook_fn = utils.create_analyse_hook_fns(self.folded_counter, _print=self.print_output)
        self.checker = Checker(all_abs_tol=abs_tol, tolerate_bias=True)

    def prepare_and_fold_model(
        self,
        use_native_rms: bool = False
    ):
        with torch.no_grad():
            with utils.HookManager(self.model, self.folded_hook_fn, self.folded_hook_pre_fn):
                self.model(self.meta_input_ids)

            for layer in self.folded_counter.center_modules:
                modules.center_modules(layer)

            if use_native_rms:
                rms = modules.native_rms_norm_forward
            else:
                rms = modules.rms_norm_forward

            for layer in self.folded_counter.layernorms:
                modules.replace_layer_norm_forward(
                    layer,
                    forward_fn=rms,
                    class_name='RMSNorm'
                )

            if self.print_output:
                print('LayerNorm:', self.folded_counter.ln_cnt)
                print('Foldable:', self.folded_counter.foldable_cnt)
                print('Center modules:', self.folded_counter.center_modules)

            return self.model

    def _check_close_and_replace(
        self,
        tensor_a: CheckItem,
        tensor_b: CheckItem,
        checker: Checker,
    ):
        checker.hide_val()
        equal, bias = checker.check_eq(tensor_a, tensor_b, abs_tol=self.abs_tol)
        if (
            equal
            and self.replace
            and isinstance(tensor_a, torch.Tensor)
            and isinstance(tensor_b, torch.Tensor)
            and bias is None
        ):
            tensor_b.data = tensor_a.data
        checker.show_val()

    def _apply_func_to_nested_tuple_pair(
        self,
        t1: CheckItem,
        t2: CheckItem,
        func,
        *args,
        **kwargs
    ):
        if isinstance(t1.value, tuple) and isinstance(t2.value, tuple):
            return tuple(
                self._apply_func_to_nested_tuple_pair(
                    CheckItem(x1, t1.name),
                    CheckItem(x2, t2.name),
                    func,
                    *args,
                    **kwargs
                )
                for x1, x2 in zip(t1.value, t2.value)
            )
        else:
            return func(t1, t2, *args, **kwargs)

    def hook_original(self, module, input, output):
        name = module.__class__.__name__
        self.output_queue.append((output, name))

    def hook_folded(self, module, input, output):
        original_output, original_name = self.output_queue.pop(0)
        original_name += ' (original)'
        folded_name = module.__class__.__name__ + ' (folded)'
        original = CheckItem(original_output, original_name)
        folded = CheckItem(output, folded_name)
        self._apply_func_to_nested_tuple_pair(
            original,
            folded,
            self._check_close_and_replace,
            self.checker,
        )

    def run_equality_checks(self):
        with torch.no_grad():
            check_input = torch.randint(0, 100, (1, 10)).cuda()
            with utils.HookManager(
                self.ref_model,
                self.hook_original,
                None,
                list(self.ref_model.modules())[1:]
            ):
                original_out = self.ref_model(check_input)

            with utils.HookManager(
                self.model,
                self.hook_folded,
                None,
                list(self.model.modules())[1:]
            ):
                folded_out = self.model(check_input)

            original_out_item = CheckItem(original_out[0], 'model output (original)')
            folded_out_item = CheckItem(folded_out[0], 'model output (folded)')

            self.checker.check_eq(
                original_out_item,
                folded_out_item,
                abs_tol=self.abs_tol,
                always_print_diff=True
            )
            self.checker.summary()

    def run_speed_tests(self):
        index = [
            "count",
            "node_id",
            "is_async",
            "is_remote",
            "use_device",
            "cpu_time_total",
            "device_time_total",
            "self_cpu_time_total",
            "self_device_time_total",
            "input_shapes",
            "stack",
            "scope",
            "cpu_memory_usage",
            "device_memory_usage",
            "self_cpu_memory_usage",
            "self_device_memory_usage",
            "cpu_children",
            "cpu_parent",
            "device_type",
            "is_legacy",
            "flops",
        ]

        for _ in range(self.speed_iter):
            json_result = {}
            current_name = ""

            def export_json(prof):
                stats = prof.key_averages()
                json_result[current_name] = {}
                for stat in stats:
                    json_result[current_name][stat.key] = {}
                    for i in index:
                        data = str(eval(f"stat.{i}"))
                        json_result[current_name][stat.key][i] = data

            my_schedule = schedule(
                wait=100,
                warmup=50,
                active=250,
            )

            start_time = time.strftime("%Y%m%d-%H%M%S")

            torch.cuda.empty_cache()
            with torch.no_grad():
                current_name = 'folded'
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    profile_memory=True,
                    record_shapes=True,
                    schedule=my_schedule,
                    with_flops=True,
                    on_trace_ready=export_json
                ) as prof:
                    for _ in range(400):
                        self.model(self.input_ids)
                        prof.step()

                current_name = 'original'
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    profile_memory=True,
                    record_shapes=True,
                    schedule=my_schedule,
                    with_flops=True,
                    on_trace_ready=export_json
                ) as prof:
                    for _ in range(400):
                        self.ref_model(self.input_ids)
                        prof.step()

            model_name = self.model.__class__.__name__
            folder = os.path.join("results", "check", model_name, "speed")
            os.makedirs(folder, exist_ok=True)
            file_path = os.path.join(folder, f"{start_time}.json")
            with open(file_path, "w") as f:
                json.dump(json_result, f, indent=4)
