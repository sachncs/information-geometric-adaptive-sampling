# igasgd

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/example/igasgd/actions/workflows/ci.yml/badge.svg)](https://github.com/example/igasgd/actions/workflows/ci.yml)

Pure Python implementation of the Drift Variation Score (DVS) adaptive sampler for graph diffusion models.

Reproduces the training-free adaptive sampling algorithms from:

> **Information-Geometric Adaptive Sampling for Graph Diffusion**
> [arXiv:2605.00250](https://arxiv.org/abs/2605.00250)

## Features

- **Algorithm 1:** DVS-driven adaptive sampler (meta-algorithm)
- **Algorithm 2:** DVS-Euler--Maruyama adaptive sampler
- **Algorithm 3:** DVS-Heun adaptive sampler (predictor-corrector)
- **Hyperparameters:** Common (Table 6) and dataset-specific (Table 7) configurations for GruM and GDSS
- **Noise schedules:** Linear, cosine, and polynomial parameterizations
- **Simplified model approximations:** Educational stand-in denoising networks (clearly marked as approximations)
- **Zero runtime dependencies:** Pure Python standard library implementation
- **Comprehensive test suite:** 153 tests covering mathematical correctness, edge cases, and numerical stability

## Installation

```bash
git clone https://github.com/example/igasgd.git
cd igasgd
pip install -e ".[dev]"
```

### From source (without dev dependencies)

```bash
pip install -e .
```

## Usage

### Quick Start

```python
from igasgd import (
    CommonConfig,
    get_dataset_config,
    DVSSampler,
    LinearSchedule,
    GruMApproximation,
    make_drift_function,
)

# Use the simplified approximation as a stand-in for a real denoising network
approx = GruMApproximation(num_nodes=9, feature_dim=4, seed=42)
drift = make_drift_function(approx)

# Load configuration from the paper
common = CommonConfig()
dataset = get_dataset_config("GruM", "QM9")

# Build and run the sampler
sampler = DVSSampler(
    drift_function=drift,
    noise_schedule=LinearSchedule(sigma_min=0.01, sigma_max=0.5),
    common_config=common,
    dataset_config=dataset,
    solver="Euler",
    seed=42,
)

# Initial noise
features_0 = [[0.1 for _ in range(4)] for _ in range(9)]
adjacency_0 = [[0.1 for _ in range(9)] for _ in range(9)]

# Sample a graph
features_t, adjacency_t, info = sampler.sample(
    initial_features=features_0,
    initial_adjacency=adjacency_0,
    terminal_time=1.0,
)

print(f"Steps: {info['total_steps'][0]}, Final time: {info['final_time'][0]}")
```

### CLI Demo

```bash
# Euler solver with GruM approximation
python examples/demo.py --model GruM --dataset QM9 --solver Euler --use-approximation

# Heun solver with GDSS approximation
python examples/demo.py --model GDSS --dataset QM9 --solver Heun --use-approximation

# With custom seed for reproducibility
python examples/demo.py --model GruM --dataset QM9 --solver Euler --use-approximation --seed 42
```

## Configuration

### Hyperparameters

**Common (Table 6):**

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| EMA Smoothing | `alpha` | 0.2 | Exponential moving average coefficient |
| Sensitivity Exponent | `beta` | 0.5 | Controls DVS sensitivity to timestep changes |
| Base Timestep | `dt_base` | 1e-3 | Reference timestep size |
| Minimum Timestep | `dt_min` | 2e-4 | Lower bound for adaptive timestep |
| Maximum Timestep | `dt_max` | 5e-3 | Upper bound for adaptive timestep |
| Stability Constant | `eps_num` | 1e-12 | Numerical stability floor |
| Boundary Tolerance | `eps_bound` | 1e-6 | Tolerance for active range boundaries |

**Dataset-specific (Table 7):**

Use `get_dataset_config(model, dataset)` to retrieve pre-configured parameters for supported model/dataset pairs (e.g., `("GruM", "QM9")`, `("GDSS", "ZINC250K")`).

### Noise Schedules

```python
from igasgd import LinearSchedule, CosineSchedule, PolynomialSchedule

# Linear interpolation
schedule = LinearSchedule(sigma_min=0.01, sigma_max=0.5)

# Cosine schedule
schedule = CosineSchedule(sigma_min=0.01, sigma_max=0.5)

# Polynomial schedule
schedule = PolynomialSchedule(sigma_min=0.01, sigma_max=0.5, exponent=2.0)
```

## Project Structure

```
igasgd/
├── src/igasgd/          # Package source code
│   ├── __init__.py      # Public API exports
│   ├── config.py        # Hyperparameter dataclasses (Tables 6 & 7)
│   ├── sampler.py       # Core DVS sampler (Algorithms 1-3)
│   ├── models.py        # Simplified denoiser approximations
│   ├── schedule.py      # Noise schedule callables
│   ├── utils.py         # Clipping, active ranges, decoding
│   └── py.typed         # PEP 561 type marker
├── tests/               # Test suite (153 tests)
├── examples/            # Runnable demos with CLI
├── docs/                # Extended documentation
│   ├── ARCHITECTURE.md  # System design and data flow
│   ├── API_REFERENCE.md # Complete public API docs
│   ├── DEPLOYMENT.md    # Production deployment guide
│   ├── DEVELOPER_GUIDE.md # Dev workflow, testing, linting
│   ├── MATH.md          # Equation-by-equation math docs
│   ├── USAGE.md         # Step-by-step usage guide
│   └── EXTENSIONS.md    # Optional enhancements roadmap
├── pyproject.toml       # Build configuration
├── LICENSE              # MIT License
├── CHANGELOG.md         # Version history
├── CONTRIBUTING.md      # Contribution guidelines
├── CODE_OF_CONDUCT.md   # Community standards
└── SECURITY.md          # Security policy
```

## Development

### Prerequisites

- Python 3.10 or later
- pip

### Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the test suite
python -m pytest tests/

# Run tests with coverage
python -m pytest tests/ --cov=igasgd --cov-report=term-missing

# Lint with ruff
ruff check src/ tests/

# Format with ruff
ruff format src/ tests/

# Type check with mypy
mypy src/igasgd

# Run the demo
python examples/demo.py --model GruM --dataset QM9 --solver Euler --use-approximation
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Build system | setuptools (via pyproject.toml) |
| Testing | pytest |
| Linting | ruff |
| Type checking | mypy |
| CI/CD | GitHub Actions |

**Runtime dependencies:** None (pure standard library).

## Roadmap

- [ ] Adaptive epsilon scheduling
- [ ] Soft clipping with smooth transitions
- [ ] NumPy vectorized operations for performance
- [ ] Numba JIT compilation support
- [ ] Structured logging with configurable verbosity
- [ ] Matplotlib visualization utilities
- [ ] Pluggable solver registry
- [ ] Synthetic drift test suites
- [ ] Regression test harness
- [ ] Additional model architecture approximations

See [docs/EXTENSIONS.md](docs/EXTENSIONS.md) for detailed descriptions of planned enhancements.

## Known Limitations

1. **Denoising networks:** The actual GruM/GDSS architectures and pretrained weights are not publicly released. The included approximations satisfy the drift interface but do **not** perform real message passing or graph attention.
2. **Noise schedule g(t):** The exact functional form used in the paper is not explicitly stated. Common parameterizations are provided; users can supply custom callables.
3. **Evaluation metrics:** FCD and NSPDK metrics require RDKit and graph-kernel libraries, which are outside the scope of this pure Python reproduction.
4. **Training code:** Out of scope; this is a sampler-only reproduction.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our development process, coding standards, and how to submit pull requests.

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Security

For reporting security vulnerabilities, please see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Provided for research and educational purposes. Please cite the original paper when using this work.

## Citation

```bibtex
@article{igasgd2026,
  title={Information-Geometric Adaptive Sampling for Graph Diffusion},
  journal={arXiv preprint arXiv:2605.00250},
  year={2026}
}
```
