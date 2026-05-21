import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.funcs.evaluate import run_evaluate_file
from src.funcs.metrics import format_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--instruction_path", type=str, default="assets/instruction_question.txt")
    parser.add_argument("--eval_recomp_ratio", "--ratio", dest="eval_recomp_ratio", type=float, default=0.3)
    parser.add_argument("--eval_recomp_method", "--method", dest="eval_recomp_method", type=str, default="vanilla", choices=["vanilla", "random", "epic", "cacheblend"])
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.0)
    args = parser.parse_args()
    if args.tau < 0:
        raise ValueError("--tau must be non-negative.")
    return args


def main():
    args = parse_args()
    summary = run_evaluate_file(args)
    print(format_summary(summary, nlp=args.nlp))


if __name__ == "__main__":
    main()
