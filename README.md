# HIJACKKV: New Threat in Position-Independent KV Cache Reuse

[[ArXiv]](https://arxiv.org/abs/2607.19957)

This is the official repository of [\[USENIX Security 2026\] HIJACKKV: New Threat in Position-Independent KV Cache Reuse](https://www.usenix.org/conference/usenixsecurity26/presentation/zhang-yichi). This paper 介绍了研究了Position-Independent KV Cache Reuse导致的安全问题，并设计了一种攻击方法来揭露这种危害。

<!-- TODO 这里放出三个 可复现Badge，稍微小一点-->

```
@inproceedings{zhang2026hijackkv,
  author = {Yichi Zhang and Zhiqi Wang and Huan Zhang and Yuchen Yang},
  title = {{HijackKV}: New Threat in {Position-Independent} {KV} Cache Reuse},
  booktitle = {35th USENIX Security Symposium (USENIX Security 26)},
  year = {2026},
}
```

### Quick Started

我们使用 [uv](https://docs.astral.sh/uv/) 管理我们的环境，可以使用以下指令进行环境同步
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --locked
```

`scripts/demo.sh` 可以用于快速开始一个 50样本的小规模实验，


```sh
bash scripts/demo.sh
```
The script attacks the first 50 samples of `data/datasets/hotpotqa_200.json` with `RATIO=0.1` and the `vanilla` method, then re-evaluates that one run with `vanilla`, `random`, `epic`, and `cacheblend`. Edit `DATASET`, `DEVICE`, `RATIO`, or `METHODS` inside the script if your local GPU layout or experiment settings differ.

| Component | Configuration |
| --- | --- |
| GPU | 4 x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB each |
| NVIDIA driver | 580.105.08 |
| CPU | 2 x AMD EPYC 9334 32-Core Processor |

## Run an Attack

Use `src/run_attack.py` to run an attack.

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


**Attack Options**

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
| `--output_dir` | `.results` | Parent directory for attack outputs. |
| `--dtype` | `bfloat16` | One of `bfloat16`, `float16`, `float32`. |
| `--instruction_path` | `data/prompts/instruction_question.txt` | System prompt file. |
| `--cache_ratio` | `0.3` | Fraction of the context tokens reused as the poisoned KV chunk. |
| `--chunk_size` | `32` | Token chunk granularity for locating that cache region. |
| `--prefix_chunk_num` | `1` | Adversarial prefix length, in chunks (`prefix_chunk_num * chunk_size` tokens). |
| `--max_new_tokens` | `5` | Tokens generated when decoding answers. |
| `--tau` | `0.0` | Decoding temperature; `0` is greedy. |

`random`, `epic`, and `cacheblend` support:
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen3-8B` (thinking disabled in chat templates)
- `mistralai/Ministral-8B-Instruct-2410`

**GCG Search Options**

| Option | Default | Description |
| --- | --- | --- |
| `--num_steps` | `250` | GCG iterations per sample. |
| `--search_width` | `256` | Candidate prefixes sampled per step. |
| `--eval_batch_size` | `128` | Maximum rows per candidate microbatch. |
| `--topk` | `512` | Per-position candidate tokens taken from the gradient. |
| `--n_replace` | `1` | Prefix positions replaced per candidate. |
| `--seed` | `0` | Base seed. Each sample derives its own stream from this seed and its `id`. |
| `--deterministic` | off | Pin every CUDA kernel to a reproducible algorithm. |
| `--early_stop_patience` | `0` (off) | Stop after this many consecutive steps whose greedy next token is already the target. |
| `--gcg_keep_best` | off | Also score the current prefix against the candidates and keep it when no candidate beats it. |

**Attack Output**

Attack runs are written to:

```text
.results/<dataset>_<gcg_method><gcg_ratio>_<device_tag>_<timestamp>/
```

The directory contains:

```text
metrics.json       # Merged attack payload
loss.png           # Loss curves, when available
parts/             # Per-device partial results
```

Each result record keeps the original sample fields and adds outputs such as
`benign_answer`, `malicious_answer`, `gcg_loss`, and `gcg_prefix_ids`, plus
`steps_run`, `hit_step` (first step whose greedy next token was the target, or
`null`), and `hit_curve` (that flag per step). 

## Run an Evaluation

Use `src/run_evaluate.py` to run an evaluation for a saved attack
payload.

```sh
uv run python src/run_evaluate.py \
  --dataset .results/<experiment_dir>
```

`--dataset` can point to either:

- an experiment directory containing `metrics.json`, or
- a specific JSON result file.

**Evaluation Options**

| Option | Default | Description |
| --- | --- | --- |
| `--dataset` | required | Experiment directory or result JSON file. |
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | Hugging Face causal LM used for re-evaluation. |
| `--device` | `0` | CUDA device id. |
| `--eval_recomp_ratio`, `--ratio` | `0.3` | Recompute ratio during evaluation. |
| `--eval_recomp_method`, `--method` | `vanilla` | One of `vanilla`, `random`, `epic`, `cacheblend`. |
| `--seed` | `0` | Seeds the evaluation-time Random recomputation draw. |
| `--deterministic` | off | Pin CUDA kernels to reproducible algorithms. |
| `--dtype` | `bfloat16` | One of `bfloat16`, `float16`, `float32`. |
| `--instruction_path` | `data/prompts/instruction_question.txt` | System prompt file. |
| `--max_new_tokens` | `5` | Tokens generated when decoding answers. |
| `--tau` | `0.0` | Decoding temperature; `0` is greedy. |

**Evaluation Output**

Each evaluation run writes a new file under:

```text
.results/<experiment_dir>/eval/
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

## Dataset

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

The repository includes four datasets, each with 200 samples:

```text
data/datasets/hotpotqa_200.json
data/datasets/squad_200.json
data/datasets/medqa_200.json
data/datasets/pubmedqa_200.json
```