import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.funcs.attack import HijackKV, run_attack_sample
from src.funcs.evaluate import run_evaluate_sample
from src.funcs.metrics import load_tokenizer, summarize_records
from src.utils import cleanup_gpu, dump_json, release_gpu_memory, save_loss_curve_plot


def parse_args():
    parser = argparse.ArgumentParser(description="Run HijackKV and generate attack results")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--instruction_path", type=str, default="assets/instruction_question.txt")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_ratio", type=float, default=0.3)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--gcg_recomp_ratio", type=float, default=0.1)
    parser.add_argument("--gcg_recomp_method", type=str, default="vanilla", choices=["vanilla", "random", "epic", "cacheblend"])
    parser.add_argument("--prefix_chunk_num", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=250)
    parser.add_argument("--n_replace", type=int, default=1)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--search_width", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--eval_recomp_ratio", type=float, default=None)
    parser.add_argument("--eval_recomp_method", type=str, default=None, choices=["vanilla", "random", "epic", "cacheblend"])
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default=".data/results")
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be a positive integer.")
    if args.tau < 0:
        raise ValueError("--tau must be non-negative.")
    if args.eval_recomp_ratio is None:
        args.eval_recomp_ratio = args.gcg_recomp_ratio
    if args.eval_recomp_method is None:
        args.eval_recomp_method = args.gcg_recomp_method

    device_ids = [item.strip() for item in args.device.split(",") if item.strip()]
    args.device = [dev if dev.startswith("cuda:") else f"cuda:{dev}" for dev in device_ids]
    args.device_tag = "".join(dev.split(":")[-1] for dev in args.device)
    args.prefix_length = args.prefix_chunk_num * args.chunk_size
    args.timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    args.result_stem = Path(args.dataset).stem
    args.dataset_name = args.result_stem.split("_")[0]
    args.recomp_tag = f"{args.gcg_recomp_method}{args.gcg_recomp_ratio:g}"
    args.experiment_name = f"{args.dataset_name}_{args.recomp_tag}_{args.device_tag}_{args.timestamp}"
    args.experiment_dir = str(Path(args.output_dir) / args.experiment_name)
    with open(args.instruction_path, "r", encoding="utf-8") as f:
        args.system_prompt = f.read().strip()
    return args


def _run_tasks(rank, device, tasks, total_tasks, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map=device,
    ).eval()
    attacker = HijackKV(model=model, tokenizer=tokenizer, device=device, args=args)

    result = []
    partial_path = Path(args.experiment_dir) / "parts" / f"{device.replace(':', '_')}.json"

    try:
        for task in tasks:
            task_idx, sample = task

            attack_result = run_attack_sample(
                attacker=attacker,
                sample=sample,
                args=args,
            )

            malicious_answer = run_evaluate_sample(
                attacker=attacker,
                prefix_ids=attack_result["best_ids"],
                tokenizer=tokenizer,
                args=args,
            )

            print(
                f"[Worker {rank}] {device} | "
                f"{task_idx + 1}/{total_tasks} | id={sample['id']} | "
                f"loss={attack_result['loss']:.4f} | step={attack_result['best_step'] + 1}"
            )

            result.append({
                "id": sample["id"],
                "context": sample["context"],
                "question": sample["question"],
                "answer": sample["answer"],
                "benign_answer": attack_result["benign_answer"],
                "target": sample["target"],
                "malicious_answer": malicious_answer,
                "gcg_loss": attack_result["loss"],
                "gcg_prefix_txt": tokenizer.decode(attack_result["best_ids"].squeeze(0), skip_special_tokens=False),
                "gcg_prefix_ids": attack_result["best_ids"][0].tolist(),
                "selected_topk_mean": attack_result["selected_topk_mean"],
                "selected_topk_var": attack_result["selected_topk_var"],
                "_loss_curve": attack_result["loss_curve"],
            })

            with open(partial_path, "w", encoding="utf-8") as f:
                dump_json(result, f, ensure_ascii=False, indent=2)
    finally:
        cleanup_gpu(device)


def _process_worker(rank, device, records, next_index, index_lock, total_tasks, args_dict):
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))

    def tasks():
        while True:
            with index_lock:
                task_idx = next_index.value
                next_index.value += 1
            if task_idx >= len(records):
                return
            yield task_idx, records[task_idx]

    _run_tasks(rank, device, tasks(), total_tasks, SimpleNamespace(**args_dict))


def main():
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "parts").mkdir(parents=True, exist_ok=True)
    interrupted = False

    try:
        with open(args.dataset, "r", encoding="utf-8") as f:
            payload = json.load(f)
        records = payload if isinstance(payload, list) else payload.get("result", [])
        records = [sample for sample in records if (sample.get("target") or "").strip()]
        if args.max_samples is not None:
            records = records[:args.max_samples]
        total_tasks = len(records)

        if len(args.device) == 1:
            _run_tasks(
                rank=0,
                device=args.device[0],
                tasks=enumerate(records),
                total_tasks=total_tasks,
                args=args,
            )
        else:
            ctx = mp.get_context("spawn")
            next_index = ctx.Value("i", 0)
            index_lock = ctx.Lock()
            processes = [
                ctx.Process(
                    target=_process_worker,
                    args=(rank, device, records, next_index, index_lock, total_tasks, vars(args).copy()),
                )
                for rank, device in enumerate(args.device)
            ]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join()
                    if process.exitcode != 0:
                        raise RuntimeError(f"Worker exited unexpectedly with exit code={process.exitcode}")
            except KeyboardInterrupt:
                interrupted = True
                print("\nInterrupt received. Terminating workers, then merging completed parts and cleaning up GPUs.")
                for process in processes:
                    if process.is_alive():
                        process.terminate()
            finally:
                for process in processes:
                    if process.is_alive():
                        process.join(timeout=1)
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=1)
                        if process.is_alive():
                            process.kill()
                    process.join(timeout=1)
                    process.close()
                del next_index
                del index_lock
                gc.collect()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        release_gpu_memory()

        merged = []
        for device in args.device:
            partial_path = experiment_dir / "parts" / f"{device.replace(':', '_')}.json"
            if partial_path.exists():
                with open(partial_path, "r", encoding="utf-8") as f:
                    merged.extend(json.load(f))
        merged.sort(key=lambda item: item["id"])
        loss_curves = [item.pop("_loss_curve", []) for item in merged]
        serializable_args = vars(args).copy()
        serializable_args.pop("system_prompt", None)
        metric_path = experiment_dir / "metrics.json"
        if any(loss_curves):
            save_loss_curve_plot(
                [curve for curve in loss_curves if curve],
                experiment_dir / "loss.png",
            )

        summary = summarize_records(merged, tokenizer=load_tokenizer(args.model))
        final_payload = {
            "metric": summary,
            "args": serializable_args,
            "result": merged,
        }
        with open(metric_path, "w", encoding="utf-8") as f:
            dump_json(final_payload, f, ensure_ascii=False, indent=2)

        if interrupted:
            print(f"Partial attack results saved to {metric_path}\n")
        else:
            print(f"Attack results saved to {metric_path}\n")


if __name__ == "__main__":
    main()
