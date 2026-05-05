# FastMatrix Chain

A production-oriented, reproducible research toolkit for multiplying compatible chains of dense matrices. The project separates **algorithmic planning** from **numerical execution**:

- Dynamic programming computes an exact minimum-scalar-multiplication parenthesization.
- NumPy executes CPU operations through its configured numerical libraries.
- An optional PyTorch backend supports CPU or CUDA tensors.
- A shape-keyed LRU cache reuses plans for repeated workloads.
- Tests, benchmarks, a notebook, Docker packaging, and CI support team use.

> Research scope: this project optimizes the order of a chain of conventional dense matrix multiplications. It does not claim a novel asymptotic matrix multiplication algorithm. Comparisons must be made on representative shapes, hardware, precision, and library builds.

## Research Questions

1. How much does minimum-FLOP parenthesization reduce estimated arithmetic work relative to left-to-right execution?
2. When does reduced arithmetic translate into lower wall-clock latency?
3. How do CPU and GPU backends compare after accounting for warm-up, synchronization, transfer, and precision?
4. When do memory layout and intermediate matrix size outweigh nominal operation count?

## Design

Given matrices `A0...An-1` with dimensions `p0 x p1`, `p1 x p2`, ..., `pn-1 x pn`, the planner evaluates every binary split:

```text
cost[i,j] = min(cost[i,k] + cost[k+1,j] + p[i] * p[k+1] * p[j+1])
```

Planning uses `O(n^3)` time and `O(n^2)` space for `n` matrices. Execution uses the resulting binary tree and delegates each `@` operation to the selected array backend.

NumPy's `multi_dot` is the primary CPU reference because its documentation states that it uses optimal parenthesization. This project implements an explicit, inspectable plan so experiments can report order and estimated scalar multiplications.

## Repository Layout

```text
.
├── src/fastmatrix/          # Installable library and benchmark CLI
├── tests/                   # Correctness, validation, and cache tests
├── notebooks/               # Executable research walkthrough
├── benchmarks/              # Script entry point
├── .github/workflows/       # CPU CI
├── pyproject.toml           # Package metadata and dependency groups
├── requirements*.txt        # pip installation entry points
├── environment.yml          # conda environment
├── Dockerfile               # Minimal CPU benchmark image
├── Makefile                 # Common development commands
└── LICENSE                  # MIT license
```

## Prerequisites

- Python 3.10 or newer
- A C/C++ runtime compatible with the installed numerical wheels
- Optional: NVIDIA GPU, compatible driver, and a suitable PyTorch build for CUDA execution

## Traditional Setup with `venv`

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

The base library alone can be installed with:

```bash
python -m pip install -e .
```

## Conda Setup

```bash
conda env create -f environment.yml
conda activate fastmatrix-chain
python -m pytest
```

## JupyterLab

After installing `requirements.txt` or creating the conda environment:

```bash
python -m ipykernel install --user --name fastmatrix-chain --display-name "Python (FastMatrix Chain)"
python -m jupyter lab
```

Open `notebooks/research_walkthrough.ipynb` and select **Python (FastMatrix Chain)** if the environment is not selected automatically.

## Optional GPU Setup

The convenience dependency is:

```bash
python -m pip install -r requirements-gpu.txt
```

PyTorch installation varies by operating system and accelerator. For controlled research environments, install the build appropriate to the team's hardware using official PyTorch guidance, then install this project without replacing that build:

```bash
python -m pip install -e ".[notebook,test]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Do not interpret GPU results unless the report confirms a `torch:cuda` backend. GPU timings must synchronize before stopping the timer; the included benchmark does this.

## Library Usage

```python
import numpy as np
from fastmatrix import multiply_chain

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

print(result.shape)
print(plan.parenthesization())
print(plan.scalar_multiplications)
```

Backend behavior:

- `numpy`: always returns a NumPy array.
- `torch`: returns a PyTorch tensor on the requested or automatically selected device.
- `auto`: selects CUDA when PyTorch is installed and CUDA is available; otherwise NumPy.

For a single input matrix, the function returns a numerically equivalent backend array after dtype and device normalization.

## Benchmarking

Run the packaged CLI:

```bash
fastmatrix-benchmark \
  --shapes 2000x80,80x1500,1500x64,64x1200 \
  --backend numpy \
  --dtype float32 \
  --repeats 10 \
  --seed 42
```

Or run the module directly:

```bash
python -m fastmatrix.benchmark --backend auto
```

The command emits versioned JSON suitable for logs and experiment tracking. Record the output with environment metadata:

```bash
mkdir -p benchmark-results
python -m fastmatrix.benchmark > benchmark-results/run.json
python -m pip freeze > benchmark-results/packages.txt
python -c "import numpy as np; np.show_config()" > benchmark-results/numpy-config.txt
```

### Benchmark Protocol

- Use fixed shapes, dtype, seed, and repeat count.
- Run one unmeasured warm-up iteration.
- Synchronize asynchronous GPU execution.
- Report minimum, median, and maximum time.
- Record CPU, GPU, RAM, operating system, Python version, package versions, and NumPy build configuration.
- Separate host-to-device transfer measurement from compute-only experiments when that distinction matters.
- Compare equivalent precision and numerical tolerances.
- Run multiple independent processes for publication-quality analysis.
- Avoid unrelated workloads and dynamic power-policy changes during measurement.

The CLI currently measures end-to-end calls, including backend normalization and any host-to-device conversion. For steady-state GPU studies, retain tensors on the GPU before timing and use the plan API to design a compute-only experiment.

## Correctness and Numerical Reproducibility

Matrix multiplication is associative in exact arithmetic, but floating-point evaluation order can change rounding. Therefore:

- Validate with `numpy.testing.assert_allclose`, not exact equality.
- Set tolerances according to dtype, scale, conditioning, and research requirements.
- Prefer `float64` for stricter numerical work when performance and memory permit.
- Report precision controls explicitly when using accelerator-specific modes.
- Do not treat matching shapes or lower latency as evidence of equivalent numerical quality.

Run quality checks:

```bash
python -m ruff check .
python -m pytest --cov=fastmatrix --cov-report=term-missing
python -m mypy src
```

## Docker

Build and run the CPU benchmark image:

```bash
docker build -t fastmatrix-chain:0.1.0 .
docker run --rm fastmatrix-chain:0.1.0 --backend numpy --repeats 5
```

GPU container execution is intentionally not embedded because driver and CUDA compatibility are deployment-specific. Add a pinned accelerator base image only after validating the team's target platform.

## CI/CD

The included GitHub Actions workflow:

- Tests Python 3.10 and 3.12.
- Runs Ruff static checks.
- Executes the unit suite with coverage.
- Does not assert performance thresholds on shared CI runners because noisy infrastructure can create misleading regressions.

For dedicated benchmark hardware, add a separate scheduled pipeline that stores JSON artifacts and compares distributions against an approved baseline.

## Known Constraints

- Dense two-dimensional matrices only.
- No sparse matrix planner.
- No distributed or multi-GPU execution.
- The planner minimizes scalar multiplication count, not measured latency or peak intermediate memory.
- Host-to-device conversion can dominate small GPU workloads.
- Input mutation during execution is outside the supported contract.
- Extremely long chains use recursive execution and may require an iterative executor.

## Research Extension Roadmap

- Add a Pareto planner balancing FLOPs, peak intermediate memory, and empirical kernel latency.
- Cache benchmark-calibrated costs by hardware, dtype, shape, and layout.
- Add JAX and CuPy adapters behind the same plan interface.
- Add sparse and structured matrix strategies.
- Add out-of-core execution with explicit memory budgets.
- Add property-based testing for randomized compatible shape chains.
- Export OpenTelemetry metrics for service deployment.

A future-state implementation could treat hardware calibration as an event-driven process. New benchmark results would update a versioned cost model, allowing subsequent executions to select plans using observed platform behavior rather than arithmetic count alone.

## Security and Governance

- The package requires no credentials or network access at runtime.
- Do not commit research datasets, proprietary matrices, secrets, or benchmark artifacts containing sensitive metadata.
- Review third-party dependencies through the team's software-composition-analysis process.
- Pin and approve versions for regulated or validated environments.
- Retain benchmark configuration and source revision with every reported result.

## References

- NumPy `multi_dot`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.multi_dot.html
- PyTorch `matmul`: https://pytorch.org/docs/stable/generated/torch.matmul.html
- PyTorch CUDA semantics: https://pytorch.org/docs/stable/notes/cuda.html
- NVIDIA cuBLAS: https://docs.nvidia.com/cuda/cublas/index.html
- JupyterLab installation: https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html

## License

MIT. See `LICENSE`.

## Contributing

1. Create a branch from the approved default branch.
2. Add or update tests with every behavior change.
3. Run lint, type checking, and tests locally.
4. Document benchmark methodology for performance claims.
5. Submit a pull request with scope, evidence, risks, and rollback considerations.
