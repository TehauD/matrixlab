from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import numpy as np

from .core import multiply_chain


def parse_shapes(value: str) -> list[tuple[int, int]]:
    try:
        shapes = [tuple(map(int, block.lower().split("x"))) for block in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use ROWSxCOLS,ROWSxCOLS") from exc
    if any(len(shape) != 2 for shape in shapes):
        raise argparse.ArgumentTypeError("Each shape must be ROWSxCOLS")
    return shapes  # compatibility is validated by the library


def synchronize(value: Any) -> None:
    if hasattr(value, "is_cuda") and value.is_cuda:
        import torch
        torch.cuda.synchronize()


def run(shapes: list[tuple[int, int]], backend: str, dtype: str, repeats: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    matrices = [rng.standard_normal(shape).astype(dtype) for shape in shapes]
    output, plan = multiply_chain(matrices, backend=backend, dtype=dtype, return_plan=True)
    synchronize(output)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        output = multiply_chain(matrices, backend=backend, dtype=dtype)
        synchronize(output)
        durations.append((time.perf_counter_ns() - started) / 1e9)
    actual_backend = f"torch:{output.device}" if hasattr(output, "device") else "numpy:cpu"
    return {
        "schema_version": "1.0", "backend": actual_backend, "dtype": dtype,
        "shapes": shapes, "plan": plan.parenthesization(),
        "scalar_multiplications": plan.scalar_multiplications, "seed": seed,
        "repeats": repeats, "median_seconds": statistics.median(durations),
        "minimum_seconds": min(durations), "maximum_seconds": max(durations),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproducible dense matrix-chain benchmark")
    parser.add_argument(
        "--shapes", type=parse_shapes,
        default=parse_shapes("2000x80,80x1500,1500x64,64x1200"),
    )
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print(json.dumps(run(args.shapes, args.backend, args.dtype, args.repeats, args.seed), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
