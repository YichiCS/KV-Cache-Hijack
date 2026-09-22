import gc
import json
import os
from pathlib import Path
from typing import Any, TextIO

import torch

def request_determinism():
    """Set the cuBLAS workspace before any CUDA context exists.

    Must run before the first cuBLAS handle is created, so call it while parsing
    arguments rather than next to the model.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def enable_determinism(seed=0):
    """Pin every kernel this process will run to a reproducible algorithm.

    Attention forward is already bit-exact, but SDPA backward reduces with
    atomics: repeated gradients for one prefix differ by ~1-3% of their scale,
    which reshuffles the top-k proposals and makes runs diverge. The deterministic
    path costs about 40% of the (batch-1) gradient pass, under 2% of a GCG step.
    """
    request_determinism()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def print_gpu_memory(device):
    device_id = int(device.split(":")[-1])
    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)
    max_allocated = torch.cuda.max_memory_allocated(device_id)
    print(f"=== {device} Memory Status ===")
    print(f"Allocated: {allocated / 1024**3:.2f} GB")
    print(f"Reserved : {reserved / 1024**3:.2f} GB")
    print(f"Peak     : {max_allocated / 1024**3:.2f} GB")
    print("==============================")


def cleanup_gpu(device):
    try:
        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    except Exception as e:
        print(f"Error cleaning up GPU {device}: {e}")


def release_gpu_memory():
    if not torch.cuda.is_available():
        return
    tensors = [torch.empty(1, device=f"cuda:{i}") for i in range(torch.cuda.device_count())]
    del tensors
    gc.collect()
    torch.cuda.empty_cache()


def _render_json(value: Any, level: int = 0, indent: int = 2, ensure_ascii: bool = False) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        pad = " " * (level * indent)
        child_pad = " " * ((level + 1) * indent)
        items = [
            f'{child_pad}{json.dumps(key, ensure_ascii=ensure_ascii)}: '
            f'{_render_json(item, level + 1, indent, ensure_ascii)}'
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"

    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(item, (dict, list)) for item in value):
            return json.dumps(value, ensure_ascii=ensure_ascii)
        pad = " " * (level * indent)
        child_pad = " " * ((level + 1) * indent)
        items = [f"{child_pad}{_render_json(item, level + 1, indent, ensure_ascii)}" for item in value]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"

    return json.dumps(value, ensure_ascii=ensure_ascii)


def dump_json(value: Any, fp: TextIO, *, ensure_ascii: bool = False, indent: int = 2) -> None:
    fp.write(_render_json(value, indent=indent, ensure_ascii=ensure_ascii))
    fp.write("\n")


def save_loss_curve_plot(curves, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for curve in curves:
        plt.plot(curve, linewidth=2)
    plt.title("Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return str(output_path)


__all__ = [
    "cleanup_gpu",
    "enable_determinism",
    "dump_json",
    "print_gpu_memory",
    "release_gpu_memory",
    "request_determinism",
    "save_loss_curve_plot",
]
