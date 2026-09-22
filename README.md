## HIJACKKV: New Threat in Position-Independent KV Cache Reuse

<p>
  <img src="assets/available.png" alt="Artifact Available" height="64">
  <img src="assets/functional.png" alt="Artifact Functional" height="64">
  <img src="assets/reproduced.png" alt="Results Reproduced" height="64">
</p>

[[ArXiv]](https://arxiv.org/abs/2607.19957)

This is the official repository for [[USENIX Security 2026] HIJACKKV: New Threat in Position-Independent KV Cache Reuse](https://www.usenix.org/conference/usenixsecurity26/presentation/zhang-yichi). This paper investigates the security risks introduced by **Position-Independent KV Cache Reuse** and presents an attack method to demonstrate these vulnerabilities.


- [Quick Start](#quick-start)
- [Tested Environment](#tested-environment)
- [Run an Attack](#run-an-attack)
- [Run an Evaluation](#run-an-evaluation)
- [Dataset](#dataset)
- [Citation](#citation)


## Quick Start

We use [uv](https://docs.astral.sh/uv/) to manage the environment.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh # Install uv
uv sync --locked # Synchronize the environment
``` 

`scripts/demo.sh` provides a quick way to run a small-scale experiment on 50 samples:

```sh
bash scripts/demo.sh
```

The script attacks the first 50 samples of `data/datasets/hotpotqa_200.json` with `RATIO=0.1` and the `vanilla` method, then re-evaluates that run using `vanilla`, `random`, `epic`, and `cacheblend`. Edit `DATASET`, `DEVICE`, `RATIO`, or `METHODS` inside the script if your local GPU configuration or experimental settings differ.

## Tested Environment

Our experiments were validated on the following machine. Exact hardware is not required, although GPU memory requirements may vary depending on the model and experimental configuration.

| Component     | Configuration                                                                |
| ------------- | ---------------------------------------------------------------------------- |
| GPU           | 4 × NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB each |
| NVIDIA driver | 580.105.08                                                                   |
| CPU           | 2 × AMD EPYC 9334 32-Core Processor                                          |

## Run an Attack

Use `src/run_attack.py` to run an attack.

```sh
uv run python src/run_attack.py \
  --dataset path/to/dataset.json \
  --device 0
```

For multiple GPUs, pass a comma-separated list of device IDs:

```sh
uv run python src/run_attack.py \
  --dataset path/to/dataset.json \
  --device 0,1
```

One worker process is launched for each listed GPU. Samples are distributed across the workers, and the partial results are merged into the final attack output after all workers finish.

**Attack Options**

| Option                 | Default                                 | Description                                                                   |
| ---------------------- | --------------------------------------- | ----------------------------------------------------------------------------- |
| `--dataset`            | required                                | Path to the input JSON dataset.                                               |
| `--device`             | `0`                                     | CUDA device ID or comma-separated device IDs.                                 |
| `--model`              | `meta-llama/Llama-3.1-8B-Instruct`      | Hugging Face causal language model.                                           |
| `--gcg_recomp_ratio`   | `0.1`                                   | Recomputation ratio during GCG optimization.                                  |
| `--gcg_recomp_method`  | `vanilla`                               | One of `vanilla`, `random`, `epic`, or `cacheblend`.                          |
| `--eval_recomp_ratio`  | same as `--gcg_recomp_ratio`            | Recomputation ratio for the post-attack evaluation pass.                      |
| `--eval_recomp_method` | same as `--gcg_recomp_method`           | Recomputation method for the post-attack evaluation pass.                     |
| `--max_samples`        | `200`                                   | Maximum number of samples with non-empty targets to process.                  |
| `--output_dir`         | `.results`                              | Parent directory for attack outputs.                                          |
| `--dtype`              | `bfloat16`                              | One of `bfloat16`, `float16`, or `float32`.                                   |
| `--instruction_path`   | `data/prompts/instruction_question.txt` | Path to the system prompt file.                                               |
| `--cache_ratio`        | `0.3`                                   | Fraction of context tokens reused as the poisoned KV cache chunk.             |
| `--chunk_size`         | `32`                                    | Token chunk granularity used to locate the cache region.                      |
| `--prefix_chunk_num`   | `1`                                     | Adversarial prefix length in chunks (`prefix_chunk_num * chunk_size` tokens). |
| `--max_new_tokens`     | `5`                                     | Number of tokens generated when decoding answers.                             |
| `--tau`                | `0.0`                                   | Decoding temperature; `0` corresponds to greedy decoding.                     |

**GCG Search Options**

| Option                  | Default   | Description                                                                                         |
| ----------------------- | --------- | --------------------------------------------------------------------------------------------------- |
| `--num_steps`           | `250`     | Number of GCG iterations per sample.                                                                |
| `--search_width`        | `256`     | Number of candidate prefixes sampled per step.                                                      |
| `--eval_batch_size`     | `128`     | Maximum number of rows per candidate microbatch.                                                    |
| `--topk`                | `512`     | Number of candidate tokens selected per position based on the gradient.                             |
| `--n_replace`           | `1`       | Number of prefix positions replaced per candidate.                                                  |
| `--seed`                | `0`       | Base random seed. Each sample derives its own random stream from this seed and its `id`.            |
| `--deterministic`       | off       | Forces CUDA kernels to use reproducible algorithms.                                                 |
| `--early_stop_patience` | `0` (off) | Stop after this many consecutive steps in which the greedy next token already matches the target.   |
| `--gcg_keep_best`       | off       | Also score the current prefix against the candidates and retain it if no candidate performs better. |

**Attack Output**

Attack runs are written to:

```text
.results/<dataset>_<gcg_method><gcg_ratio>_<device_tag>_<timestamp>/
```

The directory contains:

```text
metrics.json       # Merged attack results
loss.png           # Loss curves, when available
parts/             # Per-device partial results
```

## Run an Evaluation

Use `src/run_evaluate.py` to evaluate a saved attack result.

```sh
uv run python src/run_evaluate.py \
  --dataset .results/<experiment_dir>
```

`--dataset` can point to either:

* an experiment directory containing `metrics.json`, or
* a specific JSON result file.

**Evaluation Options**

| Option                             | Default                                 | Description                                                               |
| ---------------------------------- | --------------------------------------- | ------------------------------------------------------------------------- |
| `--dataset`                        | required                                | Experiment directory or result JSON file.                                 |
| `--model`                          | `meta-llama/Llama-3.1-8B-Instruct`      | Hugging Face causal language model used for re-evaluation.                |
| `--device`                         | `0`                                     | CUDA device ID.                                                           |
| `--eval_recomp_ratio`, `--ratio`   | `0.3`                                   | Recomputation ratio during evaluation.                                    |
| `--eval_recomp_method`, `--method` | `vanilla`                               | One of `vanilla`, `random`, `epic`, or `cacheblend`.                      |
| `--seed`                           | `0`                                     | Random seed used for the evaluation-time `random` recomputation strategy. |
| `--deterministic`                  | off                                     | Forces CUDA kernels to use reproducible algorithms.                       |
| `--dtype`                          | `bfloat16`                              | One of `bfloat16`, `float16`, or `float32`.                               |
| `--instruction_path`               | `data/prompts/instruction_question.txt` | Path to the system prompt file.                                           |
| `--max_new_tokens`                 | `5`                                     | Number of tokens generated when decoding answers.                         |
| `--tau`                            | `0.0`                                   | Decoding temperature; `0` corresponds to greedy decoding.                 |

**Evaluation Output**

Each evaluation run writes a new file under:

```text
.results/<experiment_dir>/eval/
```

The filename includes the recomputation method, ratio, and an incrementing index:

```text
metrics_tau<tau>_<method><ratio>_<index>.json
```

The evaluation output contains:

* `metric`: summary metrics
* `args`: evaluation configuration and source arguments
* `results`: updated per-sample records

## Dataset

`src/run_attack.py` expects `--dataset` to point to a JSON file. The file can be either:

* a list of sample objects, or
* an object containing a top-level `result` list.

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

The repository includes four datasets, each containing 200 samples:

```text
data/datasets/hotpotqa_200.json
data/datasets/squad_200.json
data/datasets/medqa_200.json
data/datasets/pubmedqa_200.json
```

## Citation

```bibtex
@inproceedings{zhang2026hijackkv,
  author={Yichi Zhang and Zhiqi Wang and Huan Zhang and Yuchen Yang},
  title={{HijackKV}: New Threat in {Position-Independent} {KV} Cache Reuse},
  booktitle={35th USENIX Security Symposium (USENIX Security 26)},
  year={2026},
}
```
