from typing import Optional, Any, Tuple
from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch._tensor_str import printoptions


control_codes = {
    'reset': 0,
    'bold': 1,
    'dim': 2,
    'italic': 3,
    'underline': 4,
    'reverse': 7,
    'black': 30,
    'red': 31,
    'green': 32,
    'yellow': 33,
    'blue': 34,
    'magenta': 35,
    'cyan': 36,
    'white': 37,
    'bg_black': 40,
    'bg_red': 41,
    'bg_green': 42,
    'bg_yellow': 43,
    'bg_blue': 44,
    'bg_magenta': 45,
    'bg_cyan': 46,
    'bg_white': 47,
}


def fmt_str(message: str, *formats: str) -> str:
    start = '\x1b[' + ';'.join([str(control_codes[i]) for i in formats]) + 'm'
    return start + message + '\x1b[0m'


def fail_str(message: str) -> str:
    return fmt_str(message, 'bold', 'red')


def pass_str(message: str) -> str:
    return fmt_str(message, 'bold', 'green')


def warn_str(message: str) -> str:
    return fmt_str(message, 'bold', 'yellow')


def info_str(message: str) -> str:
    return fmt_str(message, 'bold', 'blue')


def hint_str(message: str) -> str:
    return fmt_str(message, 'dim')


def _row_max_diff(
    a: Tensor, b: Tensor
) -> tuple[Tensor, Tensor, Tuple[Tensor, ...]]:
    diff = a - b
    row_diff = torch.mean(diff, dim=-1, keepdim=True)
    max_diff, max_index = torch.max(diff.abs().flatten(), 0)
    max_index_unraveled = torch.unravel_index(max_index, diff.shape)
    return row_diff, max_diff, max_index_unraveled


@dataclass
class CheckItem:
    value: Tensor | Parameter | int | float | Any
    name: str


class Checker:
    def __init__(
        self,
        print_val=True,
        all_rel_tol=None,
        all_abs_tol=None,
        tolerate_bias=True
    ) -> None:
        self.count = 0
        self.history: list[tuple[
            str,                # name of item_a
            str,                # name of item_b
            bool,               # equality result
            float,              # relative tolerance
            float,              # absolute tolerance
            Optional[Tensor]    # mean difference if bias tolerated
        ]] = []
        self.print_val = print_val
        self.all_rel_tol = all_rel_tol
        self.all_abs_tol = all_abs_tol
        self.tolerate_bias = tolerate_bias

    def hide_val(self) -> None:
        self.print_val = False

    def show_val(self) -> None:
        self.print_val = True

    def set_all_tol(
        self,
        rel_tol: Optional[float] = None,
        abs_tol: Optional[float] = None
    ) -> None:
        self.all_rel_tol = rel_tol
        self.all_abs_tol = abs_tol

    def _check_tensor_bias(
        self,
        item_a: CheckItem,
        item_b: CheckItem,
        rel_tol: float,
        abs_tol: float,
        context: dict,
    ):
        bias_passed = False
        diff = item_a.value - item_b.value
        row_diff, max_diff, max_index_unraveled = _row_max_diff(item_a.value, item_b.value)
        mean_diff = row_diff.mean()
        context['mean_diff'] = mean_diff
        # If the values in each row of diff are equal
        if torch.allclose(row_diff, diff, rtol=rel_tol, atol=abs_tol):
            if self.tolerate_bias:
                print(pass_str(f"{item_a.name} == {item_b.name} + bias"))
                context['equal'] = True
                bias_passed = True
            else:
                print(warn_str(f"{item_a.name} == {item_b.name} + bias"))
                print(warn_str(f"Mean diff: {mean_diff}"))
        if not bias_passed:
            with printoptions(precision=4, sci_mode=True):
                print(warn_str("\n".join([
                    f"Mean diff: {mean_diff}",
                    f"Max diff: {max_diff}",
                    f"Location: {[i.item() for i in max_index_unraveled]}",
                    f"{item_a.name}: {item_a.value[max_index_unraveled]}",
                    f"{item_b.name}: {item_b.value[max_index_unraveled]}",
                ])))

    def _assert(
        self,
        message: str,
        raise_exception: bool
    ) -> None:
        if raise_exception:
            raise AssertionError(message)
        else:
            print(fail_str(message))

    def check_eq(
        self,
        item_a: CheckItem,
        item_b: CheckItem,
        raise_exception: bool = False,
        rel_tol: Optional[float] = None,
        abs_tol: Optional[float] = None,
        tolerate_bias: Optional[bool] = None,
        always_print_diff: bool = False
    ) -> tuple[bool, Optional[Tensor]]:

        a = item_a.value
        b = item_b.value
        a_str = item_a.name
        b_str = item_b.name

        context = {
            'equal': False,
            'mean_diff': None
        }

        if rel_tol is None:
            if self.all_rel_tol is not None:
                rel_tol = self.all_rel_tol
            else:
                rel_tol = 1e-5

        if abs_tol is None:
            if self.all_abs_tol is not None:
                abs_tol = self.all_abs_tol
            else:
                abs_tol = 1e-6

        if tolerate_bias is None:
            tolerate_bias = self.tolerate_bias

        print(info_str(f"# {self.count} [ Test ] {a_str} ?= {b_str}"))

        if self.print_val:
            with printoptions(precision=4, sci_mode=True):
                print(hint_str(f"=== {a_str} ==="))
                print(f"{a}")
                print(hint_str(f"=== {b_str} ==="))
                print(f"{b}")

        if (
            isinstance(item_a.value, (Tensor, Parameter)) and
            isinstance(item_b.value, (Tensor, Parameter))
        ):
            try:
                context['equal'] = torch.allclose(a, b, rtol=rel_tol, atol=abs_tol)
            except RuntimeError:
                context['equal'] = False

            if not context['equal']:
                if a.shape != b.shape:
                    print(
                        fail_str(f"Shapes of {a_str} and {b_str} are different")
                    )
                    print(fail_str(f"{a_str}.shape: {a.shape}"))
                    print(fail_str(f"{b_str}.shape: {b.shape}"))
                else:
                    self._check_tensor_bias(
                        item_a,
                        item_b,
                        rel_tol,
                        abs_tol,
                        context
                    )
        elif type(a) != type(b):
            assertion_str = (
                f"# {self.count} [ Fail ] {a_str} and {b_str} have different types\n" +
                f"{type(a) = } \n" +
                f"{type(b) = }"
            )
            self._assert(assertion_str, raise_exception)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            context['equal'] = math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        else:
            context['equal'] = a == b

        if always_print_diff:
            (
                row_diff, max_diff, max_index_unraveled
            ) = _row_max_diff(item_a.value, item_b.value)
            context['mean_diff'] = row_diff.mean()
            print(f"Mean difference: {context['mean_diff']}")
            print(f"Max difference: {max_diff} at index {max_index_unraveled}")

        if not context['equal']:
            assertion_str = f"# {self.count} [ Fail ] {a_str} != {b_str}"
            self._assert(assertion_str, raise_exception)
            self.history.append((
                a_str, b_str, False, rel_tol, abs_tol, context['mean_diff']))
        else:
            print(pass_str(f"# {self.count} [ Pass ] {a_str} == {b_str}"))
            self.history.append((
                a_str, b_str, True, rel_tol, abs_tol, context['mean_diff']))

        self.count += 1
        print()

        return context['equal'], context['mean_diff']

    def summary(self):
        print(info_str(f"==== < Summary > ===="))
        for i, (a_str, b_str, result, rel_tol, abs_tol, mean_diff) in enumerate(self.history):
            if result:
                print(
                    pass_str(f"# {i} [ Pass ]"), f"{a_str} == {b_str}",
                    hint_str(f"(rel_tol={rel_tol}, abs_tol={abs_tol}, mean_diff={mean_diff})")
                )
            else:
                print(
                    fail_str(f"# {i} [ Fail ]"), f"{a_str} != {b_str}",
                    hint_str(f"(rel_tol={rel_tol}, abs_tol={abs_tol}, mean_diff={mean_diff})")
                )
        print(f"-------------------")
        total = len(self.history)
        pass_count = sum([1 for _, _, result, _, _, _ in self.history if result])
        print(info_str(f"({pass_count}/{total}) [") + "".join([
            pass_str("=") if result else
            fail_str("X") for _, _, result, _, _, _ in self.history
        ]) + info_str("]"))
        print(info_str(f"==== </Summary > ===="))
        print()
        final_eq = self.history[-1][2]
        if final_eq:
            print("Numerical equality check passed.")
        else:
            print("Numerical equality check failed.")
        print(f"Mean difference: {self.history[-1][5]}")


if __name__ == "__main__":
    check = Checker()

    a = 1
    b = 1
    check.check_eq('a', 'b')
    check.summary()

    c = 1
    d = 2
    check.check_eq('c', 'd')

    x = torch.tensor([
        [1., 2., 3.],
        [4., 5., 6.]
    ])
    y = torch.tensor([
        [1., 2., 3.],
        [4., 5., 6.]
    ])
    check.check_eq('x', 'y')

    u = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])
    v = torch.tensor([
        [1, 2, 3],
        [4, 6, 8]
    ])
    check.check_eq('u', 'v')

    e = 1
    f = 1
    check.check_eq('e', 'f', raise_exception=True)
    check.summary()

    g = 1
    h = 2
    check.check_eq('g', 'h', raise_exception=True)
