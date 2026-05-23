# HIJACKKV: New Threat in Position-Independent KV Cache Reuse

This repository contains runnable code for the HijackKV attack and for
re-evaluating saved attack results with different KV-cache recomputation
methods.

## Environment

Python is pinned to `>=3.12,<3.13`. Dependencies are managed by `uv` and locked
in `uv.lock`.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Experiment Server

The current demo and experiments are run on the following server:

| Component | Configuration |
| --- | --- |
| GPU | 4 x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB each |
| NVIDIA driver | 580.105.08 |
| CPU | 2 x AMD EPYC 9334 32-Core Processor |

The bundled `scripts/demo.sh` is configured for this machine by default with
`DEVICE=0,1,2,3`. On a different server, adjust `DEVICE` in the script or pass
a matching `--device` list to the CLI commands.

## Repository Layout

```text
.data/
  datasets/
    hotpotqa_divaco_50.json      # 50-sample demo dataset
assets/
  instruction_question.txt      # System prompt used for attack and evaluation
scripts/
  demo.sh                       # End-to-end attack and recomputation demo
src/
  run_attack.py                 # Main attack CLI
  run_evaluate.py               # Re-evaluation CLI
  kvcache/
    picache.py                  # Position-independent cache utilities
    recomps/                    # vanilla, epic, random, cacheblend methods
pyproject.toml
uv.lock
```

## Input Dataset

`src/run_attack.py` expects `--dataset` to point to a JSON file. The file can be:

- a list of sample objects, or
- an object with a top-level `result` list.

Each sample used by the attack should contain:

```json
{
  "id": 123,
  "context": "...",
  "question": "...",
  "answer": "...",
  "target": "..."
}
```

Samples with an empty `target` are skipped. During optimization, only the first
target token is used.

The repository includes a small demo dataset:

```text
.data/datasets/hotpotqa_divaco_50.json
```

It contains 50 HotPotQA-style samples and can be used directly with `--dataset`.

## Run an Attack

```sh
uv run python src/run_attack.py \
  --dataset path/to/dataset.json \
  --device 0
```

For multiple GPUs, pass a comma-separated device list:

```sh
uv run python src/run_attack.py \
  --dataset path/to/dataset.json \
  --device 0,1
```

One worker process is launched per listed GPU.

## Demo Script

`scripts/demo.sh` runs an end-to-end example on the bundled dataset:

```sh
bash scripts/demo.sh
```

The script runs attack on `.data/datasets/hotpotqa_divaco_50.json` with
`RATIO=0.1`, then re-evaluates the generated attack with `vanilla`, `random`,
`epic`, and `cacheblend`.

Edit `DATASET`, `DEVICE`, `RATIO`, or `METHODS` inside the script if your local
GPU layout or experiment settings differ.

### Common Attack Options

| Option | Default | Description |
| --- | --- | --- |
| `--dataset` | required | Input JSON dataset path. |
| `--device` | `0` | CUDA device id or comma-separated ids. |
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | Hugging Face causal LM. |
| `--gcg_recomp_ratio` | `0.1` | Recompute ratio during GCG optimization. |
| `--gcg_recomp_method` | `vanilla` | One of `vanilla`, `random`, `epic`, `cacheblend`. |
| `--eval_recomp_ratio` | same as `--gcg_recomp_ratio` | Recompute ratio for the post-attack evaluation pass. |
| `--eval_recomp_method` | same as `--gcg_recomp_method` | Recompute method for the post-attack evaluation pass. |
| `--max_samples` | `200` | Maximum non-empty-target samples to process. |
| `--output_dir` | `.data/results` | Parent directory for attack outputs. |

`random` and `cacheblend` use model-internal attention code and currently support
Llama and Qwen3 model backends.

## Attack Output

Attack runs are written to:

```text
.data/results/<dataset>_<gcg_method><gcg_ratio>_<device_tag>_<timestamp>/
```

The directory contains:

```text
metrics.json       # Merged attack payload
loss.png           # Loss curves, when available
parts/             # Per-device partial results
```

Each result record keeps the original sample fields and adds outputs such as
`benign_answer`, `malicious_answer`, `gcg_loss`, and `gcg_prefix_ids`.

## Re-evaluate Results

Use `src/run_evaluate.py` to re-run decoding and metrics for a saved attack
payload.

```sh
uv run python src/run_evaluate.py \
  --dataset .data/results/<experiment_dir>
```

`--dataset` can point to either:

- an experiment directory containing `metrics.json`, or
- a specific JSON result file.

### Common Evaluation Options

| Option | Default | Description |
| --- | --- | --- |
| `--dataset` | required | Experiment directory or result JSON file. |
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | Hugging Face causal LM used for re-evaluation. |
| `--device` | `0` | CUDA device id. |
| `--eval_recomp_ratio`, `--ratio` | `0.3` | Recompute ratio during evaluation. |
| `--eval_recomp_method`, `--method` | `vanilla` | One of `vanilla`, `random`, `epic`, `cacheblend`. |

Each evaluation run writes a new file under:

```text
.data/results/<experiment_dir>/eval/
```

The filename includes the recomputation method, ratio, and an incrementing
index:

```text
metrics_tau<tau>_<method><ratio>_<index>.json
```

Evaluation output stores:

- `metric`: summary metrics
- `args`: evaluation configuration and source arguments
- `results`: updated per-sample records