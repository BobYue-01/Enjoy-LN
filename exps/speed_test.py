import argparse
import random
import json

import torch
import transformers
from matplotlib import pyplot as plt

from utils.analyzer import Analyzer
from utils.recorder import StatsRecorder


def model_tensor_memory(model):
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total  # bytes


def record_iters(recorder, section_name, iters, func, *args, **kwargs):
    with torch.no_grad():
        for _ in range(iters):
            with recorder.section(section_name):
                torch.cuda.synchronize()
                func(*args, **kwargs)
                torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='GPT2', help='Model name')
    parser.add_argument("--batch_size", type=int, default=1, help='Batch size')
    parser.add_argument("--seq_length", type=int, default=512, help='Sequence length')
    parser.add_argument("--warmup_iters", type=int, default=50, help='Warmup iterations')
    parser.add_argument("--run_iters", type=int, default=50, help='Run iterations')
    parser.add_argument("--img_file", type=str, default='performance.png', help='Output image file name')
    parser.add_argument("--json_file", type=str, default='metrics.json', help='Output JSON file name')
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    recorder = StatsRecorder()

    Config: transformers.PretrainedConfig = getattr(transformers, args.model_name + 'Config')
    Model: transformers.PreTrainedModel = getattr(transformers, args.model_name + 'Model')

    with recorder.section('model_init'):
        config = Config()
        config.max_position_embeddings = args.seq_length
        model = Model(config).cuda()

    input_ids = torch.randint(0, 1024, (args.batch_size, args.seq_length), requires_grad=False).cuda()

    # warm up forwards
    record_iters(recorder, 'warmup_forwards_before_folding', args.warmup_iters, model, input_ids)

    # forwards before folding
    record_iters(recorder, 'forward_before_folding', args.run_iters, model, input_ids)

    # record tensor size before folding
    model_size_before_folding = model_tensor_memory(model)

    with recorder.section('folding'):
        analyzer = Analyzer(model, print_output=False)
        model = analyzer.prepare_and_fold_model()

    model_size_after_folding = model_tensor_memory(model)

    # warm up forwards
    record_iters(recorder, 'warmup_forwards_after_folding', args.warmup_iters, model, input_ids)

    # forwards after folding
    record_iters(recorder, 'forward_after_folding', args.run_iters, model, input_ids)

    def _fmt(sec):
        records = recorder.stats.get(sec, [])
        if not records:
            return "N/A"
        total_time = sum(record.get('time', 0.0) for record in records)
        avg_time = total_time / len(records)
        return f"{avg_time:.4f} seconds"

    json.dump({
        "model": args.model_name,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "warmup_iters": args.warmup_iters,
        "time_model_init": _fmt('model_init'),
        "avg_time_forward_before": _fmt('forward_before_folding'),
        "time_folding": _fmt('folding'),
        "avg_time_forward_after": _fmt('forward_after_folding'),
        "mem_model_before": model_size_before_folding / 1024**2,
        "mem_model_after": model_size_after_folding / 1024**2,
        "time_forward_before": recorder.stats.get('forward_before_folding', []),
        "time_forward_after": recorder.stats.get('forward_after_folding', []),
    }, open(args.json_file, "w"), indent=4)

    sections = [
        'warmup_forwards_before_folding',
        'forward_before_folding',
        'warmup_forwards_after_folding',
        'forward_after_folding'
    ]
    times = []
    for sec in sections:
        records = recorder.stats.get(sec, [])
        times.extend(record.get('time', 0.0) for record in records)
    plt.plot(times, marker='o')
    plt.title(f'Performance Comparison for {args.model_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(args.img_file)


if __name__ == "__main__":
    with torch.no_grad():
        main()
