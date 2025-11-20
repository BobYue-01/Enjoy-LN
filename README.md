# Enjoy-LN: Efficient Layer Normalization Folding Tool for Large Language Models

## Environment Requirements

Although Enjoy-LN is designed to be broadly compatible across platforms and PyTorch/Transformers versions, we recommend running the tool under the following configurations to ensure stable compilation and reproducible results:

- **PyTorch ≥ 2.2**
- **Transformers ≥ 4.34**
- **Linux environment (recommended for building CUDA/C++ kernels)**

Other environments may work, but the above configuration has been tested in our evaluations.

> [!IMPORTANT]
>
> Enjoy-LN currently supports inference only.
>
> The tool performs LayerNorm folding for efficient forward passes but does not yet support training or backpropagation through folded modules.

## Installation Guide

### 1. Clone the repository

```bash
git clone https://github.com/EnjoyYourLN/Enjoy-LN.git
cd Enjoy-LN
```

### 2. Install the package in editable mode

```bash
pip install -e . --no-build-isolation
```

### 3. Build and install optimized C++ / CUDA RMSNorm kernels

Enter the CUDA implementation directory and install:

```bash
cd rms_norm_kernels/rms_norm_cuda
pip install . --no-build-isolation
```

We also offer a C++ implementation. It is optional, as our main results are based on the CUDA version. To install the C++ version:

```bash
cd ../rms_norm_cpp
pip install . --no-build-isolation
```

> [!Tip]
>
> To uninstall the package, run:
>
> ```bash
> pip uninstall enjoyln rms-norm-cuda rms-norm-cpp -y
> ```

## Running Experiments

Our folding tool supports all LLMs predefined in Hugging Face `transformers.{ModelName}Model`.

You can specify the model using: `--model_name {ModelName}`

For example, GPT-2 is defined in transformers as `GPT2Config` and `GPT2Model`. Thus, you can simply specify: `--model_name GPT2`

### 1. Verifying numerical equality

To check whether a model can be folded and to evaluate the numerical equality before and after folding:

```bash
python exps/verification.py
```

Results are saved in `results/verification/<model_name>/`.

- `structure.txt` shows the model architecture and, in the last three lines, the number of folded LN modules and centered linear layers.
- `equality.txt` reports numerical comparison results (last two lines).

> [!IMPORTANT]
>
> In the unit tests of our custom RMSNorm CUDA kernel, we observed a sudden increase in numerical error when the `hidden_size` exceeds a certain threshold (e.g., 499 in our testing environment).
>
> This behavior is specific to our current RMSNorm CUDA implementation and does **not** affect the LayerNorm folding method itself.
>
> By default, verification uses the native PyTorch RMSNorm implementation to test the correctness of `prepare_and_fold_model`. Under this setting, the numerical discrepancy of the final model output remains below 1e-7, which is sufficient to validate the core claims of the paper.
>
> To validate with our custom kernel, run:
>
> ```bash
> python exps/verification.py --use_fast_rms --hidden_size 120   # hidden_size > 120 may cause error spikes
> ```
>
> We are actively investigating the cause of the kernel-level numerical instability and will provide an updated kernel once the issue is fully diagnosed. This does not influence the primary results or conclusions presented in the paper.

### 2. Comparing speed

To compare the forward-pass speed before and after folding:

```bash
python exps/speed_test.py
```

We also provide a batch benchmark script:

```bash
python exps/batch_speed_test.py
```

Results are saved under `results/performance/<model_name>/<bs>_<len>/run_<time>`.

`metrics.json` includes:

- `time_model_init`: model initialization time
- `time_forward_before`: forward time before folding
- `time_folding`: folding time
- `time_forward_after`: forward time after folding

The speed experiments currently use our custom RMSNorm implementation as a reference.

We hope the community can help develop an official, faster RMSNorm kernel in the future.
