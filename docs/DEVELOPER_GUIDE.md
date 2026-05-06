# Developer Guide

This guide covers how to build, test, lint, and extend the `igasgd` codebase.

---

## 1. Development Setup

```bash
# Clone or navigate to the repository
cd /path/to/igasgd

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
python -m pip install -e ".[dev]"
```

---

## 2. Running Tests

The project uses a lightweight custom test runner that works without pytest (for maximum portability), but pytest is also supported via `pyproject.toml` configuration.

### Without pytest

```bash
# Run each test file individually
python tests/test_sampler.py
python tests/test_config.py
python tests/test_models.py
python tests/test_schedule.py
python tests/test_utils.py

# Or run all in a loop
for f in tests/test_*.py; do python "$f"; done
```

### With pytest

```bash
# Run the entire suite
python -m pytest

# Run with coverage
python -m pytest --cov=src/igasgd --cov-report=term-missing

# Run a specific test class
python -m pytest tests/test_sampler.py::TestDVSSamplerEndToEnd -v
```

### Current Test Count

| File | Tests |
|------|-------|
| `test_sampler.py` | 70 |
| `test_config.py` | 15 |
| `test_models.py` | 23 |
| `test_schedule.py` | 18 |
| `test_utils.py` | 27 |
| **Total** | **153** |

---

## 3. Linting and Type Checking

```bash
# Lint with ruff
ruff check src/ tests/

# Type check with mypy
mypy src/igasgd
```

The `pyproject.toml` configures:
- `ruff` with Google-style docstrings (`D` rules) and import sorting (`I`).
- `mypy` with `disallow_untyped_defs` and `disallow_incomplete_defs`.

---

## 4. Code Style

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

Key conventions:
- `snake_case` for functions, methods, variables.
- `PascalCase` for class names.
- `UPPER_CASE` for module-level constants.
- Line length at or below 100 characters.
- Docstrings for all public modules, classes, methods, and functions.
- Type hints everywhere.

---

## 5. Adding a New Test

1. Pick the file whose behaviour you are testing (`test_sampler.py`, `test_models.py`, etc.).
2. Add a new method inside the appropriate `Test*` class:
   ```python
   def test_my_new_behaviour(self) -> None:
       result = my_function(input)
       assert result == expected
   ```
3. Run the file directly to verify:
   ```bash
   python tests/test_sampler.py
   ```

If you add a new test file, make sure it includes the `sys.path.insert` boilerplate so it can be run standalone:

```python
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src"))
```

---

## 6. Adding a New Solver

1. Implement the step function in `sampler.py`:
   ```python
   def my_solver_step(
       features, adjacency, drift_features, drift_adjacency,
       timestep, noise_scale, drift_function, time, rng
   ):
       # Your logic here
       return next_features, next_adjacency
   ```

2. Add the solver branch in `DVSSampler.sample()`:
   ```python
   if self._solver == "MySolver":
       next_features, next_adjacency = my_solver_step(...)
   ```

3. Update `DVSSampler.__init__` to accept the new solver name:
   ```python
   if solver not in {"Euler", "Heun", "MySolver"}:
       raise ValueError(...)
   ```

4. Add tests in `test_sampler.py` and update the `__init__.py` exports if needed.

---

## 7. Adding a New Noise Schedule

Create a callable class in `schedule.py`:

```python
class MySchedule:
    def __init__(self, param: float) -> None:
        self._param = param

    def __call__(self, time: float) -> float:
        return time ** self._param
```

Then pass it to `DVSSampler` like any other schedule. No other changes are required.

---

## 8. Adding a New Dataset Configuration

Add a row to `DATASET_CONFIGS` in `config.py`:

```python
("GruM", "MyDataset"): DatasetConfig(
    model="GruM", dataset="MyDataset",
    kappa_ref=2.0,
    gamma_euler=0.15, gamma_heun=0.18,
    active_range=[(0.0, 1.0)],
),
```

Add a lookup test in `test_config.py` and a functional test in `test_sampler.py` if needed.

---

## 9. Adding a New Model Approximation

Subclass `SimpleGraphDenoiser` in `models.py`:

```python
class MyModelApproximation(SimpleGraphDenoiser):
    def __init__(self, num_nodes: int, feature_dim: int, seed: int = 42) -> None:
        super().__init__(num_nodes, feature_dim, hidden_dim=64, seed=seed)
```

Add tests in `test_models.py` and update `__init__.py` exports.

---

## 10. Working with the Demo

The demo script lives in `examples/demo.py`. It is a self-contained CLI that exercises the full pipeline.

```bash
# Quick smoke test
python examples/demo.py --model GruM --dataset QM9 --solver Euler --use-approximation

# Verbose output
python examples/demo.py --model GDSS --dataset QM9 --solver Heun --seed 7
```

---

## 11. Package Structure

```
igasgd/
├── src/igasgd/          — Source code
│   ├── __init__.py      — Public API
│   ├── config.py        — Hyperparameters
│   ├── sampler.py       — Core algorithms
│   ├── models.py        — Stand-in networks
│   ├── schedule.py      — Noise schedules
│   └── utils.py         — Helpers
├── tests/               — Test suite (153 tests)
│   ├── test_sampler.py
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_schedule.py
│   └── test_utils.py
├── examples/            — CLI demo
│   └── demo.py
├── docs/                — Documentation
│   ├── INDEX.md
│   ├── MATH.md
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── USAGE.md
│   ├── DEVELOPER_GUIDE.md
│   └── EXTENSIONS.md
├── pyproject.toml       — Modern packaging
├── MANIFEST.in          — Source distribution manifest
├── .gitignore           — Git ignore rules
├── LICENSE              — MIT License
├── CONTRIBUTING.md      — Contribution guidelines
├── CHANGELOG.md         — Version history
├── README.md            — Project overview
├── FIDELITY_REPORT.md  — Paper comparison
└── REPRODUCTION_SUMMARY.md — Technical summary
```

---

## 12. Release Checklist

Before cutting a release:

1. Run the full test suite: `python -m pytest`
2. Run linting: `ruff check src/ tests/`
3. Run type checking: `mypy src/igasgd`
4. Run the demo: `python examples/demo.py --use-approximation`
5. Verify no forbidden words in the codebase:
   ```bash
   grep -riE "TODO|FIXME|stub|mock|placeholder|dummy|pass" src/ tests/ examples/ || echo "Clean"
   ```
6. Update `CHANGELOG.md` and `pyproject.toml` version.
7. Build the package:
   ```bash
   python -m build
   ```
8. Verify the sdist contains all required files:
   ```bash
   tar -tzf dist/igasgd-*.tar.gz | grep -E "docs|tests|examples"
   ```
