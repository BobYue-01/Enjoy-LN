import itertools
import subprocess
import os
from datetime import datetime

CUDA_VISIBLE_DEVICES = "0"

# Define experiment parameters
MODELS = ["GPT2", "Bert", "Bloom", "OPT", "Phi", "ViT"]
BATCH_SIZES = [1, 2]
SEQ_LENGTHS = [256, 1024, 4096, 16384]

# Repeats for each experiment configuration
REPEATS = 5

# Warmup iterations before measuring performance
WARMUP_ITERS = 50
RUN_ITERS = 50

RESULT_DIR = "results"

out_dir = os.path.join(RESULT_DIR, "performance")

for model_name, bs, seqlen in itertools.product(
        MODELS, BATCH_SIZES, SEQ_LENGTHS):

    group_folder = os.path.join(out_dir, model_name, f"bs{bs}_len{seqlen}")
    os.makedirs(group_folder, exist_ok=True)

    for _ in range(REPEATS):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        run_folder = os.path.join(group_folder, f"run_{timestamp}")
        os.makedirs(run_folder, exist_ok=True)

        out_img = os.path.join(run_folder, "performance.png")
        out_json = os.path.join(run_folder, "metrics.json")

        print(f"\n=== Running {model_name} bs={bs} len={seqlen} warm={WARMUP_ITERS} @ {timestamp} ===")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

        cmd = [
            "python", "exps/speed_test.py",
            "--model_name", model_name,
            "--batch_size", str(bs),
            "--seq_length", str(seqlen),
            "--warmup_iters", str(WARMUP_ITERS),
            "--run_iters", str(RUN_ITERS),
            "--img_file", out_img,
            "--json_file", out_json,
        ]

        subprocess.run(cmd, env=env)
