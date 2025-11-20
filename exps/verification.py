import argparse
import random
import os
from contextlib import redirect_stdout

import torch
import transformers

from utils.analyzer import Analyzer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default='GPT2', help='Model name')
    parser.add_argument("--use_fast_rms", action="store_true", help='Use our fast RMSNorm')
    parser.add_argument("--hidden_size", type=int, default=None, help='Custom hidden size')
    parser.add_argument("--abs_tol", type=float, default=1e-5, help='Absolute tolerance')
    parser.add_argument("--replace", action="store_true", help='Replace output')
    parser.add_argument("--skip_equality", action="store_true", help='Check equality of output')
    parser.add_argument("--no_skip_speed", action="store_true", help='Speed comparison')
    parser.add_argument("--speed_iter", type=int, default=16, help='Speed iteration')

    args = parser.parse_args()

    if args.use_fast_rms and not args.hidden_size <= 120:
        print(
            "Warning: for unknown reasons, fast RMSNorm with large hidden size",
            "will cause Numerical errors that cannot be ignored."
        )

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    Config: transformers.PretrainedConfig = getattr(transformers, args.model_name + 'Config')
    Model: transformers.PreTrainedModel = getattr(transformers, args.model_name + 'Model')

    config = Config()
    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
    model = Model(config).cuda()
    ref_model = Model(config).cuda()

    analyzer = Analyzer(
        model,
        abs_tol=args.abs_tol,
        replace=args.replace,
        speed_iter=args.speed_iter,
        ref_model=ref_model
    )

    folder = os.path.join("results", "verification", args.model_name)
    os.makedirs(folder, exist_ok=True)

    structure_file = os.path.join(folder, "structure.txt")
    with open(structure_file, "w") as f:
        with redirect_stdout(f):
            analyzer.prepare_and_fold_model(
                use_native_rms=True
            )

    if not args.skip_equality:
        equality_file = os.path.join(folder, "equality.txt")
        with open(equality_file, "w") as f:
            with redirect_stdout(f):
                analyzer.run_equality_checks()

    # Speed test using `torch.profile`
    # Too complicated, thus deprecated
    if args.no_skip_speed:
        analyzer.run_speed_tests()


if __name__ == "__main__":
    with torch.no_grad():
        main()
