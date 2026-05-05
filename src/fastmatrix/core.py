from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np

Backend = Literal["auto", "numpy", "torch"]


@dataclass(frozen=True)
class Plan:
    dimensions: tuple[int, ...]
    split: tuple[tuple[int, ...], ...]
    scalar_multiplications: int

    @property
    def matrix_count(self) -> int:
        return len(self.dimensions) - 1

    def parenthesization(self) -> str:
        def build(i: int, j: int) -> str:
            if i == j:
                return f"A{i}"
            k = self.split[i][j]
            return f"({build(i, k)} @ {build(k + 1, j)})"
        return build(0, self.matrix_count - 1)


def dimensions_from_shapes(shapes: Sequence[Sequence[int]]) -> tuple[int, ...]:
    if not shapes:
        raise ValueError("At least one matrix is required.")
    normalized = [tuple(map(int, shape)) for shape in shapes]
    if any(len(shape) != 2 or min(shape) <= 0 for shape in normalized):
        raise ValueError("Every matrix must have a positive two-dimensional shape.")
    for index, (left, right) in enumerate(zip(normalized, normalized[1:], strict=False)):
        if left[1] != right[0]:
            raise ValueError(f"Incompatible shapes at {index}/{index + 1}: {left} and {right}.")
    return (normalized[0][0], *(shape[1] for shape in normalized))


@lru_cache(maxsize=256)
def plan_dimensions(dimensions: tuple[int, ...]) -> Plan:
    if len(dimensions) < 2 or min(dimensions) <= 0:
        raise ValueError("Dimensions must describe at least one positive-sized matrix.")
    n = len(dimensions) - 1
    cost: list[list[int | None]] = [[0 if i == j else None for j in range(n)] for i in range(n)]
    split = [[0 for _ in range(n)] for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            best: int | None = None
            for k in range(i, j):
                left, right = cost[i][k], cost[k + 1][j]
                assert left is not None and right is not None
                candidate = left + right + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                if best is None or candidate < best:
                    best, split[i][j] = candidate, k
            cost[i][j] = best
    return Plan(dimensions, tuple(tuple(row) for row in split), int(cost[0][n - 1] or 0))


def plan_chain(matrices: Sequence[Any]) -> Plan:
    shapes = [getattr(matrix, "shape", None) for matrix in matrices]
    if any(shape is None for shape in shapes):
        raise TypeError("Every input must expose a matrix shape.")
    return plan_dimensions(dimensions_from_shapes(shapes))


def _execute(matrices: Sequence[Any], plan: Plan, i: int, j: int) -> Any:
    if i == j:
        return matrices[i]
    k = plan.split[i][j]
    return _execute(matrices, plan, i, k) @ _execute(matrices, plan, k + 1, j)


def multiply_chain(
    matrices: Sequence[Any], *, backend: Backend = "auto", dtype: str = "float32",
    device: str | None = None, return_plan: bool = False,
) -> Any:
    """Multiply compatible dense matrices using a cached exact minimum-FLOP plan.

    Auto uses CUDA when PyTorch is installed and CUDA is available; otherwise NumPy.
    The result remains a torch.Tensor for the torch backend and ndarray for NumPy.
    """
    if backend not in ("auto", "numpy", "torch"):
        raise ValueError("backend must be auto, numpy, or torch.")
    if not matrices:
        raise ValueError("At least one matrix is required.")

    torch = None
    if backend in ("auto", "torch"):
        try:
            import torch as imported_torch
            torch = imported_torch
        except ImportError:
            if backend == "torch":
                raise RuntimeError("Install the optional gpu dependencies to use torch.") from None

    use_torch = torch is not None and (backend == "torch" or torch.cuda.is_available())
    if use_torch:
        target = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype, None)
        if torch_dtype is None:
            raise ValueError(f"Unsupported torch dtype: {dtype}")
        work = [torch.as_tensor(value, dtype=torch_dtype, device=target) for value in matrices]
    else:
        np_dtype = np.dtype(dtype)
        work = [np.ascontiguousarray(value, dtype=np_dtype) for value in matrices]

    plan = plan_chain(work)
    result = _execute(work, plan, 0, len(work) - 1)
    return (result, plan) if return_plan else result
