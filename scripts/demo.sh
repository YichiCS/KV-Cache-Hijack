set -euo pipefail

DATASET="/home/ymz5721/HijackKV/KV-Cache-Hijack/.data/datasets/hotpotqa_divaco_50.json"
OUTPUT_DIR=".data/results"
RATIO=0.1
METHODS=(vanilla random epic cacheblend)
DEVICE=0,1,2,3


uv run python src/run_attack.py --dataset "$DATASET" --device $DEVICE --max_samples 50 --gcg_recomp_ratio "$RATIO" --gcg_recomp_method vanilla --output_dir "$OUTPUT_DIR"

EXPERIMENT_DIR=$(
  find "$OUTPUT_DIR" -maxdepth 1 -type d -name "hotpotqa_vanilla${RATIO}_0123_*" -printf '%T@ %p\n' |
    sort -nr |
    head -n 1 |
    cut -d' ' -f2-
)

for METHOD in "${METHODS[@]}"; do
  uv run python src/run_evaluate.py --dataset "$EXPERIMENT_DIR" --device 0 --eval_recomp_ratio "$RATIO" --eval_recomp_method "$METHOD"
done
