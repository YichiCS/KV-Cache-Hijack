# HIJACKKV: New Threat in Position-Independent KV Cache Reuse

This repository contains the current runnable code for the HijackKV attack and
for re-evaluating saved attack results.

The code focuses on two workflows:

1. Run a HijackKV attack on an existing JSON dataset.
2. Re-evaluate a saved attack result with a chosen cache recomputation method.

Dataset generation scripts are not included in the current repository snapshot.
Prepare the input JSON dataset before running the attack.

## Environment

Python is pinned to `>=3.12,<3.13`. Dependencies are managed by `uv`.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

The direct project dependencies are declared in `pyproject.toml`:

- `torch==2.10.0`
- `transformers==5.1.0`
- `accelerate==1.12.0`
- `tqdm==4.67.3`
- `matplotlib==3.10.8`

The lock file keeps the resolved transitive dependencies reproducible.

## Repository Layout

```text
assets/
  instruction_question.txt      # System prompt used for attack and evaluation
src/
  run_attack.py                 # Main attack CLI
  run_evaluate.py               # Re-evaluation CLI
  funcs/
    attack.py                   # HijackKV optimization loop
    evaluate.py                 # Evaluation helpers
    metrics.py                  # TASR/UASR metrics
  kvcache/
    picache.py                  # Position-independent cache utilities
    recomps/                    # vanilla, epic, random, cacheblend methods
  utils/
    llm.py                      # Prompting, KV-cache, decoding helpers
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

Notes:

- Samples with an empty `target` are skipped.
- `target` is tokenized and only the first target token is optimized against.
- `context` must be present inside the prompt assembled from
  `assets/instruction_question.txt`, `context`, and `question`.
- The context should be long enough to contain at least one full cache chunk
  under `--chunk_size`; otherwise cache splitting cannot proceed.

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

### Attack Options

| Option | Default | Description |
| --- | --- | --- |
| `--dataset` | required | Input JSON dataset path. |
| `--device` | `0` | CUDA device id or comma-separated ids. |
| `--instruction_path` | `assets/instruction_question.txt` | System prompt file. |
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | Hugging Face causal LM. |
| `--cache_ratio` | `0.3` | Fraction of context chunks reused as PI cache. |
| `--chunk_size` | `32` | Token chunk size for cache slicing. |
| `--max_new_tokens` | `5` | Greedy decoding length for benign/malicious answers. |
| `--tau` | `0.0` | Decoding temperature; `0` means greedy argmax. |
| `--gcg_recomp_ratio` | `0.1` | Recompute ratio during GCG optimization. |
| `--gcg_recomp_method` | `vanilla` | One of `vanilla`, `random`, `epic`, `cacheblend`. |
| `--prefix_chunk_num` | `1` | Number of optimized prefix chunks. |
| `--num_steps` | `250` | GCG optimization steps. |
| `--n_replace` | `1` | Tokens replaced per sampled candidate. |
| `--topk` | `512` | Top-k token candidates selected from gradients. |
| `--search_width` | `256` | Number of sampled candidates per step. |
| `--eval_batch_size` | `128` | Batch size for candidate loss evaluation. |
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

`metrics.json` has this top-level shape:

```json
{
  "metric": {
    "num_samples": 0,
    "num_valid": 0,
    "tasr": 0.0,
    "uasr": 0.0,
    "counts": {
      "tasr": 0,
      "uasr": 0
    }
  },
  "args": {},
  "result": []
}
```

Each result record keeps the original sample fields and adds attack outputs such
as:

- `benign_answer`
- `malicious_answer`
- `gcg_loss`
- `gcg_prefix_txt`
- `gcg_prefix_ids`
- `selected_topk_mean`
- `selected_topk_var`

If an attack is interrupted, completed per-device partial files under `parts/`
are still merged into `metrics.json` during cleanup.

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

### Evaluation Options

| Option | Default | Description |
| --- | --- | --- |
| `--dataset` | required | Experiment directory or result JSON file. |
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | Hugging Face causal LM used for re-evaluation. |
| `--device` | `0` | CUDA device id. |
| `--instruction_path` | `assets/instruction_question.txt` | System prompt file. |
| `--eval_recomp_ratio`, `--ratio` | `0.3` | Recompute ratio during evaluation. |
| `--eval_recomp_method`, `--method` | `vanilla` | One of `vanilla`, `random`, `epic`, `cacheblend`. |
| `--max_new_tokens` | `5` | Decoding length. |
| `--tau` | `0.0` | Decoding temperature; `0` means greedy argmax. |

Each evaluation run writes a new file under:

```text
.data/results/<experiment_dir>/eval/
```

The filename includes the temperature, recomputation method, ratio, and an
incrementing index:

```text
metrics_tau<tau>_<method><ratio>_<index>.json
```

Evaluation output stores:

- `metric`: summary metrics
- `args`: evaluation configuration and source arguments
- `results`: updated per-sample records

## Metrics

The repository reports two token-level attack metrics:

- `TASR`: target attack success rate, where the first malicious answer token
  matches the first target token.
- `UASR`: untargeted attack success rate, where the first malicious answer token
  differs from the first benign answer token.

If re-evaluating with a model whose tokenizer differs from the source model,
the code switches to a first-word string comparison mode for metrics.

## Prompt File

`assets/instruction_question.txt` is the active prompt file used by both attack
and evaluation. It is inserted as the system prompt before the sample `context`
and `question`.
