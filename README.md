# MatrixLab

<div align="center">

## Same Result. Different Cost.

**MatrixLab makes the hidden decisions inside matrix-chain computation visible, measurable, and reproducible.**

[![CI](https://github.com/OWNER/REPOSITORY/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPOSITORY/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/backend-NumPy-4D77CF?logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/backend-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA Optional](https://img.shields.io/badge/CUDA-optional-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

[Why MatrixLab Exists](#why-matrixlab-exists) · [The Mathematics](#the-mathematics-made-visible) · [The Surprise](#the-surprise-fewer-operations-do-not-always-run-faster) · [Quick Start](#quick-start) · [Benchmarking](#benchmarking) · [Roadmap](#research-extension-roadmap)

</div>

---

<p align="center">
  <img
    src="/assets/matrixlab-order-matters.svg"
    alt="The same three matrices require 7,500 scalar multiplications with the optimal parenthesization and 75,000 with the alternative parenthesization"
    width="100%">
</p>

## Why MatrixLab Exists

Most matrix multiplication code hides a consequential decision.

Given a chain:

```text
A0 @ A1 @ A2
```

the matrices may be multiplied in either valid order:

```text
(A0 @ A1) @ A2
```

or:

```text
A0 @ (A1 @ A2)
```

Both expressions are mathematically equivalent in exact arithmetic. Both return a matrix with the same dimensions. Yet the amount of arithmetic and the size of intermediate matrices can differ dramatically.

MatrixLab exists to expose that hidden decision.

It provides a controlled way to answer:

- Which multiplication order minimizes estimated scalar operations?
- How much arithmetic does that order avoid?
- Which intermediate matrices does the plan create?
- Does lower estimated arithmetic produce lower wall-clock latency?
- How do NumPy, PyTorch CPU, and PyTorch CUDA behave under comparable conditions?
- Which parts of the outcome come from planning, memory, transfers, synchronization, precision, or the numerical library build?

MatrixLab turns matrix-chain multiplication from an opaque operation into an inspectable research workflow:

```text
Plan the computation
        ↓
Expose the decision
        ↓
Execute the plan
        ↓
Validate the result
        ↓
Measure the behavior
        ↓
Preserve the evidence
```

> **Research scope**
>
> MatrixLab optimizes the parenthesization of compatible chains of conventional dense, two-dimensional matrix multiplications. It does **not** claim a novel asymptotic matrix multiplication algorithm. The current planner minimizes estimated scalar multiplication count under the classical matrix-chain cost model. It does not directly optimize wall-clock latency, peak memory, energy use, or numerical error.

## The Mathematics Made Visible

Consider three matrices:

```text
A0 is 10 x 100
A1 is 100 x 5
A2 is 5 x 50
```

The inner dimensions are compatible:

```text
(10 x 100) @ (100 x 5) @ (5 x 50)
```

The final result is always:

```text
10 x 50
```

The route to that result is not computationally equivalent.

### Path 1: Multiply Left First

```text
(A0 @ A1) @ A2
```

#### Step 1

```text
A0                 A1                 Intermediate
10 x 100     @     100 x 5      =     10 x 5
```

For a matrix product `(m x k) @ (k x n)`, the classical scalar multiplication estimate is:

```text
m * k * n
```

Therefore:

```text
10 * 100 * 5 = 5,000
```

A small `10 x 5` intermediate matrix is produced.

#### Step 2

```text
Intermediate        A2                Final result
10 x 5        @     5 x 50      =     10 x 50
```

The second multiplication costs:

```text
10 * 5 * 50 = 2,500
```

#### Total

```text
5,000 + 2,500 = 7,500 scalar multiplications
```

### Path 2: Multiply Right First

```text
A0 @ (A1 @ A2)
```

#### Step 1

```text
A1                 A2                 Intermediate
100 x 5      @     5 x 50      =     100 x 50
```

The first multiplication costs:

```text
100 * 5 * 50 = 25,000
```

This path creates a much larger `100 x 50` intermediate matrix.

#### Step 2

```text
A0                 Intermediate        Final result
10 x 100     @     100 x 50      =     10 x 50
```

The second multiplication costs:

```text
10 * 100 * 50 = 50,000
```

#### Total

```text
25,000 + 50,000 = 75,000 scalar multiplications
```

### Same Result, Different Cost

| Measure | `(A0 @ A1) @ A2` | `A0 @ (A1 @ A2)` |
|---|---:|---:|
| First result shape | `10 x 5` | `100 x 50` |
| First multiplication | `5,000` | `25,000` |
| Second multiplication | `2,500` | `50,000` |
| Total estimated scalar multiplications | **`7,500`** | **`75,000`** |
| Relative arithmetic work | **`1x`** | **`10x`** |
| Final result shape | `10 x 50` | `10 x 50` |

The selected plan:

- Avoids the larger `100 x 50` intermediate matrix.
- Eliminates `67,500` estimated scalar multiplications.
- Uses `10%` of the arithmetic estimated for the alternative order.
- Produces the same mathematical result shape.

This is the first reason MatrixLab exists.

## How MatrixLab Finds the Plan

<p align="center">
  <img
    src="/assets/matrixlab-plan-selection.svg"
    alt="MatrixLab evaluates matrix-chain subproblems with dynamic programming, reconstructs the minimum-cost binary tree, and caches the selected plan"
    width="100%">
</p>

Given matrices `A0 ... A(n-1)` with dimensions:

```text
A0:      p0 x p1
A1:      p1 x p2
...
A(n-1):  p(n-1) x pn
```

MatrixLab evaluates every valid binary split `k` for each subchain `[i, j]`:

```text
cost[i, j] = min(
    cost[i, k]
    + cost[k + 1, j]
    + p[i] * p[k + 1] * p[j + 1]
)
```

The planner separates a chain into overlapping subproblems, stores the lowest known cost for each subchain, and reconstructs the minimum-cost binary execution tree.

For `n` matrices:

- Planning time is `O(n^3)`.
- Planning space is `O(n^2)`.
- The output is an inspectable execution plan.
- The plan reports its parenthesization.
- The plan reports estimated scalar multiplications.
- A shape-keyed LRU cache can reuse the plan for repeated dimension signatures.

The selected plan is exact for the classical scalar-count objective. It is not necessarily optimal for latency, memory consumption, energy use, or numerical error on a particular platform.

## The Surprise: Fewer Operations Do Not Always Run Faster

A natural expectation is:

```text
fewer scalar multiplications = lower wall-clock latency
```

That expectation is not universally true.

<p align="center">
  <img
    src="/assets/matrixlab-flops-vs-runtime.svg"
    alt="A tenfold reduction in estimated arithmetic work did not produce lower latency in the displayed small exploratory workload"
    width="100%">
</p>

### Exploratory Notebook Evidence

The demonstrated operation-count example showed:

```text
75,000 operations
        ↓
7,500 operations

10x less estimated arithmetic work
```

A separate, very small `float64` timing experiment produced:

| Statistic | Optimized plan | Left-to-right |
|---|---:|---:|
| Count | 10 | 10 |
| Mean | `0.023460 ms` | `0.018240 ms` |
| Standard deviation | `0.009625 ms` | `0.002510 ms` |
| Minimum | `0.015400 ms` | `0.016700 ms` |
| Median | `0.018600 ms` | `0.017000 ms` |
| Maximum | `0.042600 ms` | `0.023600 ms` |

The optimized path did not reduce measured latency in that exploratory microbenchmark.

That result is not a contradiction. Estimated scalar count and observed runtime measure different things.

### What Runtime Also Includes

Wall-clock performance may be affected by:

- Planning and cache-lookup overhead
- Python and framework dispatch
- Array allocation and conversion
- Memory layout and contiguity
- Intermediate matrix dimensions
- CPU cache behavior and memory bandwidth
- BLAS kernel selection and threading
- GPU kernel launch overhead
- Host-to-device and device-to-host transfer
- Device synchronization
- Dtype and accelerator precision controls
- Dynamic frequency scaling
- Competing system workloads
- Timer resolution for extremely short operations

This is the second and more important reason MatrixLab exists:

> MatrixLab does not merely find a lower arithmetic cost. It creates a reproducible environment for testing whether that mathematical advantage matters on real hardware.

### Evidence Boundaries

The displayed timing observations are exploratory and should not be generalized.

The visible notebook environment included:

```text
Operating system: Windows 11 10.0.26200 SP0
Python: 3.12.10
NumPy: 2.3.3
Processor architecture: AMD64
Repeats: 10
```

Publication-quality analysis requires:

- Representative matrix shapes
- Multiple independent processes
- More observations
- Captured CPU and memory details
- Captured NumPy and BLAS configuration
- Controlled background workload and power policy
- Explicit cold-cache and warm-cache conditions
- Equivalent precision and tolerance settings
- Clear separation of planning, transfer, and compute time

## Numerical Correctness

The demonstrated `float64` experiment compared MatrixLab with `numpy.linalg.multi_dot`:

```python
rng = np.random.default_rng(42)
matrices = [
    rng.normal(size=shape)
    for shape in [(200, 20), (20, 100), (100, 10), (10, 80)]
]

actual, plan = multiply_chain(
    matrices,
    backend="numpy",
    dtype="float64",
    return_plan=True,
)

reference = np.linalg.multi_dot(matrices)

np.testing.assert_allclose(
    actual,
    reference,
    rtol=1e-12,
    atol=1e-12,
)
```

The displayed plan was:

```text
((A0 @ (A1 @ A2)) @ A3)
```

The displayed result shape was:

```text
(200, 80)
```

This confirms numerical agreement for the demonstrated chain, dtype, backend, and tolerance settings. It does not replace broader randomized, adversarial, ill-conditioned, dtype-specific, or backend-specific validation.

Floating-point multiplication is not perfectly associative. Different valid parenthesizations can produce small rounding differences even when both results are numerically acceptable.

## What MatrixLab Provides

MatrixLab separates five concerns that are often mixed together:

### 1. Planning

- Validate compatible matrix dimensions.
- Compute an exact minimum-scalar-multiplication parenthesization.
- Expose the selected binary execution tree.
- Report estimated scalar multiplications.

### 2. Reuse

- Key plans by the full matrix-dimension signature.
- Reuse plans through an LRU cache.
- Separate planning cost from repeated execution studies.

### 3. Execution

- Execute on NumPy CPU.
- Optionally execute on PyTorch CPU.
- Optionally execute on PyTorch CUDA.
- Automatically select CUDA when configured and available.

### 4. Measurement

- Warm up before measurement.
- Synchronize asynchronous GPU operations.
- Emit versioned JSON benchmark output.
- Capture environment and backend metadata.
- Support transfer-inclusive and compute-focused studies.

### 5. Reproducibility

- Validate numerical results with declared tolerances.
- Preserve matrix shapes, dtype, seed, and repeat count.
- Record source revision and dependency versions.
- Provide tests, a research notebook, Docker packaging, and CI.

## MatrixLab and NumPy `multi_dot`

NumPy `multi_dot` is the primary CPU numerical reference and already uses optimized parenthesization. MatrixLab adds explicit plans, operation-count evidence, multiple execution backends, and a reproducible research protocol.

| Capability | NumPy `multi_dot` | MatrixLab |
|---|:---:|:---:|
| Optimized matrix-chain parenthesization | Yes | Yes |
| Numerical result | Yes | Yes |
| Explicit plan object | No | Yes |
| Human-readable parenthesization | No | Yes |
| Estimated scalar multiplication count | No | Yes |
| Project-exposed shape-keyed plan cache | No | Yes |
| NumPy execution | Yes | Yes |
| Optional PyTorch CPU/CUDA execution | No | Yes |
| Benchmark CLI | No | Yes |
| Versioned JSON experiment output | No | Yes |
| Research notebook and protocol | No | Yes |
| Docker and CI workflow | No | Yes |

MatrixLab does not replace NumPy, PyTorch, BLAS, LAPACK, or cuBLAS. It makes the planning and measurement layer explicit while delegating numerical kernels to the selected backend.

## Quick Start

### Install

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Inspect a Known Plan

```python
from matrixlab import plan_dimensions

plan = plan_dimensions((10, 100, 5, 50))

print(plan.parenthesization())
print(plan.scalar_multiplications)
```

Expected output:

```text
((A0 @ A1) @ A2)
7500
```

### Execute a Matrix Chain

```python
import numpy as np
from matrixlab import multiply_chain

rng = np.random.default_rng(42)
matrices = [
    rng.standard_normal((2000, 80), dtype=np.float32),
    rng.standard_normal((80, 1500), dtype=np.float32),
    rng.standard_normal((1500, 64), dtype=np.float32),
    rng.standard_normal((64, 1200), dtype=np.float32),
]

result, plan = multiply_chain(
    matrices,
    backend="auto",
    dtype="float32",
    return_plan=True,
)

print("Result shape:", result.shape)
print("Parenthesization:", plan.parenthesization())
print("Estimated scalar multiplications:", plan.scalar_multiplications)
```

## End-to-End Workflow

<p align="center">
  <img
    src="/assets/matrixlab-workflow.svg"
    alt="Compatible matrices move through validation, exact planning, shape-keyed caching, backend execution, numerical validation, and evidence capture"
    width="100%">
</p>

```text
Compatible matrices
        ↓
Validate dimensions
        ↓
Look up shape-keyed plan
        ↓
Cache hit? ── yes ──→ reuse plan
        │
        no
        ↓
Compute exact dynamic-programming plan
        ↓
Cache plan
        ↓
Select NumPy or PyTorch backend
        ↓
Execute the binary plan
        ↓
Return result and optional plan metadata
        ↓
Validate, benchmark, and preserve evidence
```

## Backend Behavior

### `backend="numpy"`

- Executes with NumPy on CPU.
- Returns a NumPy array.
- Uses the numerical libraries configured for the installed NumPy build.

### `backend="torch"`

- Requires PyTorch.
- Returns a PyTorch tensor.
- Supports CPU tensors.
- Supports CUDA tensors when a compatible PyTorch build, driver, and device are available.

### `backend="auto"`

- Selects CUDA when PyTorch and a usable CUDA device are available.
- Otherwise selects NumPy.
- Must resolve to a recorded backend and device for comparable experiments.

For a single input matrix, MatrixLab returns a numerically equivalent backend array after applicable dtype and device normalization. No chain optimization is required.

## Repository Layout

```text
.
├── src/matrixlab/           # Planner, plan model, executors, cache, and CLI
├── tests/                   # Correctness, validation, backend, and cache tests
├── notebooks/               # Executable research walkthrough
├── benchmarks/              # Benchmark entry points
├── docs/assets/             # README infographics
├── .github/workflows/       # CPU CI workflows
├── pyproject.toml           # Package metadata and dependency groups
├── requirements*.txt        # pip installation entry points
├── environment.yml          # Conda environment
├── Dockerfile               # CPU benchmark image
├── Makefile                 # Common development commands
├── README.md                # Project narrative and operating guidance
└── LICENSE                  # MIT license
```

## Installation

### Requirements

- Python 3.10 or newer
- A compatible runtime for the installed numerical wheels
- Sufficient memory for inputs and intermediate matrices
- Optional NVIDIA GPU and compatible driver
- Optional PyTorch build appropriate for the target CUDA runtime

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

### Conda

```bash
conda env create -f environment.yml
conda activate matrixlab
python -m pytest
```

## JupyterLab

```bash
python -m ipykernel install \
  --user \
  --name matrixlab \
  --display-name "Python (MatrixLab)"

python -m jupyter lab
```

Open:

```text
notebooks/research_walkthrough.ipynb
```

Use the notebook for:

- Plan inspection
- Known operation-count examples
- Numerical validation
- Exploratory timing
- Distribution visualization
- Hypothesis development

Use the benchmark CLI for reportable experiments.

## Optional GPU Setup

```bash
python -m pip install -r requirements-gpu.txt
python -m pip install -e ".[notebook,test]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Do not describe a run as a GPU result unless the captured report confirms a `torch:cuda` backend and the intended device.

For GPU timing:

- Warm up the device.
- Synchronize asynchronous execution.
- Separate transfer-inclusive and compute-only measurements.
- Retain tensors on the GPU for steady-state compute studies.
- Record device, driver, runtime, PyTorch version, dtype, and precision controls.

## Benchmarking

### Packaged CLI

```bash
matrixlab-benchmark \
  --shapes 2000x80,80x1500,1500x64,64x1200 \
  --backend numpy \
  --dtype float32 \
  --repeats 10 \
  --seed 42
```

### Module Entry Point

```bash
python -m matrixlab.benchmark --backend auto
```

### Preserve Results

```bash
mkdir -p benchmark-results
python -m matrixlab.benchmark > benchmark-results/run.json
python -m pip freeze > benchmark-results/packages.txt
python -c "import numpy as np; np.show_config()" > benchmark-results/numpy-config.txt
```

### Benchmark Protocol

For reportable experiments:

1. Fix shapes, dtype, seed, repeat count, and backend.
2. Record the requested and resolved backend and device.
3. Run at least one unmeasured warm-up iteration.
4. Synchronize asynchronous GPU execution.
5. Distinguish end-to-end and compute-only measurements.
6. Preserve raw observations in addition to aggregates.
7. Report minimum, median, maximum, variation, and sample count.
8. Capture CPU, GPU, RAM, operating system, Python, dependencies, and NumPy configuration.
9. Compare equivalent precision and numerical tolerances.
10. Run multiple independent processes.
11. Record whether the plan cache was cold or warm.
12. State whether allocation, normalization, conversion, and transfer were included.
13. Retain the Git commit SHA and working-tree state.
14. Avoid unrelated workloads and uncontrolled power-policy changes.

The CLI currently measures end-to-end calls, including backend normalization and applicable host-to-device conversion. For steady-state GPU studies, retain tensors on the accelerator and use the plan API to isolate compute execution.

## Correctness and Reproducibility

Use tolerance-aware validation:

```python
np.testing.assert_allclose(
    actual,
    reference,
    rtol=1e-12,
    atol=1e-12,
)
```

Guidance:

- Do not require bitwise equality across plans, devices, or library builds.
- Set tolerances according to dtype, scale, conditioning, and research requirements.
- Prefer `float64` when stricter numerical agreement is required and resources permit.
- Record accelerator-specific precision controls.
- Do not treat matching output shapes as proof of numerical equivalence.
- Do not treat lower latency as proof of equivalent numerical quality.

### Quality Checks

```bash
python -m ruff check .
python -m mypy src
python -m pytest --cov=matrixlab --cov-report=term-missing
```

### Recommended Test Coverage

- Compatible and incompatible shape chains
- Empty and single-matrix inputs
- Non-two-dimensional inputs
- Parenthesization correctness
- Scalar-count correctness
- NumPy numerical correctness
- PyTorch CPU correctness
- PyTorch CUDA correctness when available
- Dtype and device normalization
- Cache hits, misses, reuse, and eviction
- Long-chain and large-shape stress cases
- Tolerance behavior across dtypes

## Docker

```bash
docker build -t matrixlab:0.1.0 .
docker run --rm matrixlab:0.1.0 --backend numpy --repeats 5
```

GPU container execution is deployment-specific. Add a pinned accelerator image only after validating driver, runtime, supply-chain, and target-platform requirements.

## CI/CD

The CPU CI workflow should:

- Test supported Python versions.
- Run Ruff and mypy.
- Execute unit tests with coverage.
- Build and install the package.
- Validate documentation links where practical.
- Avoid hard latency thresholds on shared runners.

Use dedicated benchmark hardware for scheduled performance baselines. Preserve versioned JSON artifacts and compare distributions rather than isolated timings.

## Known Constraints

- Dense, two-dimensional matrices only
- No sparse or structured matrix planner
- No distributed or multi-GPU execution
- Scalar multiplication count is the current planning objective
- No direct latency-aware cost model
- No direct peak-memory-aware objective
- Host-to-device transfer may dominate small GPU workloads
- Input mutation during execution is outside the supported contract
- Extremely long chains use recursive execution and may need an iterative executor
- Benchmark findings remain environment-specific

## Research Questions

1. How much does minimum-scalar-count parenthesization reduce estimated arithmetic work relative to left-to-right execution?
2. When does lower estimated arithmetic translate into lower wall-clock latency?
3. How do CPU and GPU backends compare after warm-up, synchronization, transfer, and precision are controlled?
4. When do memory layout and intermediate matrix dimensions outweigh nominal operation count?
5. How stable are observed differences across processes, machines, and numerical library builds?
6. When does planning, normalization, allocation, or transfer overhead become material?
7. Can empirical hardware calibration improve plan selection without sacrificing reproducibility and explainability?

## Research Extension Roadmap

### Planning

- Add a Pareto planner for arithmetic, peak intermediate memory, and empirical latency.
- Add pluggable and versioned cost models.
- Add hardware-calibrated planning.
- Add an iterative executor for extremely long chains.

### Backends

- Add JAX and CuPy adapters behind the plan interface.
- Add sparse and structured strategies.
- Explore explicit-memory-budget and out-of-core execution.

### Research Quality

- Add property-based testing for randomized compatible chains.
- Add numerical conditioning studies.
- Add independent-process benchmark orchestration.
- Add confidence intervals and effect-size reporting.
- Add standardized benchmark artifact schemas.

### Operations

- Export OpenTelemetry metrics for service deployment.
- Add regression dashboards backed by dedicated hardware.
- Add event-driven hardware calibration.
- Bind every empirical plan to a versioned cost model.

### Future State

A future MatrixLab implementation could continuously calibrate hardware-specific execution costs from controlled benchmark events. A versioned empirical model could complement the deterministic classical plan while preserving evidence lineage, auditability, and reproducibility.

## Security and Governance

- No credentials are required for local matrix-chain execution.
- No network access is required at runtime.
- Matrix values do not need to leave the local process.
- Do not commit proprietary matrices, sensitive datasets, secrets, or identifying hardware inventories.
- Review third-party dependencies through software-composition analysis.
- Pin approved versions for regulated or validated environments.
- Retain benchmark configuration, source revision, environment metadata, and raw observations.
- Use least-privilege CI permissions and protected release workflows.

## Reproducibility Checklist

Before publishing a result, capture:

- [ ] Matrix shapes and generation method
- [ ] Random seed and dtype
- [ ] Requested and resolved backend
- [ ] CPU, GPU, RAM, and device details
- [ ] Operating system and Python version
- [ ] NumPy build configuration and dependencies
- [ ] PyTorch, CUDA runtime, and driver when applicable
- [ ] Warm-up and synchronization policy
- [ ] Transfer-inclusive or compute-only scope
- [ ] Cold or warm plan-cache state
- [ ] Repeat count and independent-process count
- [ ] Raw timings and aggregation method
- [ ] Numerical tolerances and validation outcome
- [ ] Parenthesization and estimated scalar multiplication count
- [ ] Git commit SHA and working-tree state

## Troubleshooting

### The Optimized Plan Is Slower

This can be a valid result.

- Confirm whether planning was included.
- Compare cold-cache and warm-cache behavior.
- Increase workload size and repeat count.
- Run multiple independent processes.
- Separate allocation, normalization, transfer, and compute time.
- Inspect memory layout and intermediate shapes.
- Capture NumPy and BLAS configuration.

### CUDA Is Not Selected

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Verify the NVIDIA driver, PyTorch build, CUDA compatibility, and device visibility. Record the resolved backend rather than assuming CUDA was selected.

### Results Differ Slightly

- Use tolerance-aware assertions.
- Increase precision where appropriate.
- Inspect matrix scale and conditioning.
- Capture the selected plan, dtype, device, and numerical library build.

## References

- [NumPy `multi_dot`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.multi_dot.html)
- [PyTorch `matmul`](https://pytorch.org/docs/stable/generated/torch.matmul.html)
- [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)
- [JupyterLab installation](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)

## Citation

<!-- TODO: Replace author, repository URL, version, year, and DOI. -->

```bibtex
@software{matrixlab_TODO_YEAR,
  author  = {TODO: Author Name},
  title   = {MatrixLab: Making Matrix Computation Decisions Visible},
  year    = {TODO: Year},
  version = {TODO: Release Version},
  url     = {https://github.com/OWNER/REPOSITORY},
  license = {MIT},
  doi     = {TODO: DOI}
}
```

Archive and cite a versioned release for published research.

## Contributing

1. Create a branch from the approved default branch.
2. Keep changes scoped and non-regressive.
3. Add or update tests for every behavior change.
4. Run linting, type checking, and tests locally.
5. Document methodology for performance-related changes.
6. Include scope, evidence, risks, and rollback considerations in the pull request.
7. Do not merge generalized performance claims without reproducible artifacts.

## License

MatrixLab is licensed under the MIT License. See [LICENSE](LICENSE).

## Project Status

MatrixLab is an active research and engineering project. Interfaces, benchmark schemas, and backend behavior may evolve before a stable `1.0.0` release. Pin versions and retain the exact source revision for reproducible studies.
