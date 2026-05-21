import json
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.funcs.attack import HijackKV
from src.funcs.metrics import metric, same_tokenizer, summarize_records
from src.kvcache import PICacheManager, cache_concat
from src.kvcache.recomps import cache_recomputation
from src.utils import dump_json
from src.utils.llm import greedy_decode


def run_evaluate_sample(attacker, prefix_ids, tokenizer, args):
    with torch.no_grad():
        malicious_cache, _ = attacker.build_malicious_cache(prefix_ids=prefix_ids)
        recomp_cache = cache_recomputation(
            model=attacker.model,
            malicious_cache=malicious_cache,
            picm=attacker.picm,
            ratio=args.eval_recomp_ratio,
            method=args.eval_recomp_method,
        )
        return greedy_decode(
            model=attacker.model,
            tokenizer=tokenizer,
            input_ids=attacker.query_ids,
            cache=cache_concat([attacker.context_cache, recomp_cache]),
            args=args,
        )


def next_eval_metrics_path(
    eval_dir: Path,
    tau: float,
    eval_recomp_method: str,
    eval_recomp_ratio: float,
) -> Path:
    existing_indices = []
    file_prefix = f"metrics_tau{tau:g}_{eval_recomp_method}{eval_recomp_ratio:g}"
    pattern = f"{file_prefix}_*.json"

    for path in eval_dir.glob(pattern):
        suffix = path.stem.removeprefix(f"{file_prefix}_")
        if suffix.isdigit():
            existing_indices.append(int(suffix))
    next_index = max(existing_indices, default=-1) + 1
    return eval_dir / f"{file_prefix}_{next_index}.json"


def run_evaluate_file(args):
    dataset_path = Path(args.dataset)
    device = args.device if args.device.startswith("cuda:") else f"cuda:{args.device}"

    if dataset_path.is_dir():
        experiment_dir = dataset_path
        metric_path = dataset_path / "metrics.json"
        if not metric_path.exists():
            raise ValueError(f"Expected metrics.json in {dataset_path}.")
        dataset_path = metric_path
    else:
        experiment_dir = dataset_path.parent

    with open(args.instruction_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    with open(dataset_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    chunk_size = getattr(args, "chunk_size", None)
    cache_ratio = getattr(args, "cache_ratio", None)

    if isinstance(payload, list):
        records = payload
        source_args = {}
        resolved_chunk_size = chunk_size
        resolved_cache_ratio = cache_ratio
    else:
        records = payload.get("result")
        if records is None:
            records = payload.get("results", [])
        args_config = payload.get("args", {})
        source_args = args_config
        resolved_chunk_size = args_config.get("chunk_size", chunk_size)
        resolved_cache_ratio = args_config.get("cache_ratio", cache_ratio)

    source_model = source_args.get("model")
    args.nlp = not same_tokenizer(args.model, source_model) if source_model else False
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map=device,
    ).eval()

    decode_args_dict = dict(source_args)
    decode_args_dict.update(
        system_prompt=system_prompt,
        cache_ratio=resolved_cache_ratio,
        chunk_size=resolved_chunk_size,
        max_new_tokens=args.max_new_tokens,
        tau=args.tau,
        eval_recomp_ratio=args.eval_recomp_ratio,
        eval_recomp_method=args.eval_recomp_method,
    )
    
    decode_args = SimpleNamespace(**decode_args_dict)
    attacker = HijackKV(model=model, tokenizer=tokenizer, device=device, args=decode_args)

    for sample in tqdm(records, desc="Evaluating attack results", dynamic_ncols=True):
        picm = PICacheManager(sample, model, tokenizer, device, decode_args)
        benign_answer = greedy_decode(
            model=model,
            tokenizer=tokenizer,
            input_ids=picm.ids_group["query"],
            cache=picm.cache_group["full"],
            args=decode_args,
        )

        prefix_ids = torch.tensor([sample["gcg_prefix_ids"]], dtype=torch.long, device=device)
        attacker.setup(picm)
        malicious_answer = run_evaluate_sample(
            attacker=attacker,
            prefix_ids=prefix_ids,
            tokenizer=tokenizer,
            args=decode_args,
        )

        sample["benign_answer"] = benign_answer
        sample["malicious_answer"] = malicious_answer

        valid, tasr, uasr = metric(
            benign_answer=benign_answer,
            malicious_answer=malicious_answer,
            target=sample.get("target", ""),
            tokenizer=None if args.nlp else tokenizer,
            nlp=args.nlp,
        )
        eval_result = {
            "eval_recomp_method": args.eval_recomp_method,
            "eval_recomp_ratio": args.eval_recomp_ratio,
            "benign_answer": benign_answer,
            "malicious_answer": malicious_answer,
            "valid": valid,
            "TASR": tasr,
            "UASR": uasr,
        }
        sample.setdefault("eval_result", []).append(eval_result)

    summary = summarize_records(records, tokenizer=None if args.nlp else tokenizer, nlp=args.nlp)
    eval_dir = experiment_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_payload = {
        "metric": summary,
        "args": {
            "source_dataset": str(dataset_path),
            "model": args.model,
            "device": device,
            "instruction_path": args.instruction_path,
            "eval_recomp_ratio": args.eval_recomp_ratio,
            "eval_recomp_method": args.eval_recomp_method,
            "cache_ratio": resolved_cache_ratio,
            "chunk_size": resolved_chunk_size,
            "max_new_tokens": args.max_new_tokens,
            "tau": args.tau,
            "nlp": args.nlp,
            "source_args": source_args,
        },
        "results": records,
    }
    with open(
        next_eval_metrics_path(
            eval_dir,
            args.tau,
            args.eval_recomp_method,
            args.eval_recomp_ratio,
        ),
        "w",
        encoding="utf-8",
    ) as f:
        dump_json(eval_payload, f, ensure_ascii=False, indent=2)

    return summary
