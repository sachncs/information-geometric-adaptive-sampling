# Deployment Guide

This document covers deploying `igasgd` in various environments.

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/example/igasgd.git
cd igasgd
pip install -e .
```

### As a Dependency

Add to your project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "igasgd",
]

# Or pin to a version
dependencies = [
    "igasgd>=0.1.0",
]
```

### In a Requirements File

```
igasgd
```

## Environment Considerations

### Python Version

Requires Python 3.10 or later. Verify with:

```bash
python --version  # Must be >= 3.10
```

### No Runtime Dependencies

`igasgd` uses only the Python standard library. No additional system packages are required.

### Virtual Environments

Always install in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

pip install igasgd
```

## Production Usage

### Basic Integration

```python
from igasgd import (
    CommonConfig,
    get_dataset_config,
    DVSSampler,
    LinearSchedule,
    make_drift_function,
)

# Initialize once at module level
_common_config = CommonConfig()
_dataset_configs = {
    ("GruM", "QM9"): get_dataset_config("GruM", "QM9"),
    ("GDSS", "ZINC250K"): get_dataset_config("GDSS", "ZINC250K"),
}

def generate_graph(drift_function, model, dataset, seed=None):
    """Generate a graph using DVS adaptive sampling."""
    config = _dataset_configs[(model, dataset)]
    schedule = LinearSchedule(sigma_min=0.01, sigma_max=0.5)

    sampler = DVSSampler(
        drift_function=drift_function,
        noise_schedule=schedule,
        common_config=_common_config,
        dataset_config=config,
        solver="Euler",
        seed=seed,
    )

    # Initial noise (shapes must match your graph dimensions)
    features_0 = [[0.0] * 4 for _ in range(9)]
    adjacency_0 = [[0.0] * 9 for _ in range(9)]

    features_t, adjacency_t, info = sampler.sample(
        initial_features=features_0,
        initial_adjacency=adjacency_0,
        terminal_time=1.0,
    )

    return features_t, adjacency_t, info
```

### Performance Considerations

- **Numerical stability:** Use the default `eps_num=1e-12` unless you have specific requirements.
- **Solver choice:** Euler is faster; Heun is more accurate. Choose based on your accuracy/performance tradeoff.
- **Timestep bounds:** Adjust `dt_min` and `dt_max` if you observe convergence issues or excessive step counts.

### Resource Usage

| Resource | Typical Usage |
|----------|---------------|
| Memory | Proportional to graph size (nodes x features) |
| CPU | Pure Python; single-threaded. Consider NumPy vectorization for large graphs (see `docs/EXTENSIONS.md`) |
| Network | None (no external calls) |
| Disk | None (no caching) |

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy and install package
COPY . .
RUN pip install --no-cache-dir .

# Copy application code
COPY your_app/ /app/your_app/

CMD ["python", "-m", "your_app.main"]
```

### Build and Run

```bash
docker build -t your-app .
docker run your-app
```

## Testing in Production

### Smoke Test

After deployment, verify the package works:

```python
from igasgd import DVSSampler, CommonConfig, LinearSchedule

config = CommonConfig()
schedule = LinearSchedule(sigma_min=0.01, sigma_max=0.5)

# Minimal test
def dummy_drift(X, A, t):
    return [[0.0] * len(X[0]) for _ in X], [[0.0] * len(A[0]) for _ in A]

sampler = DVSSampler(
    drift_function=dummy_drift,
    noise_schedule=schedule,
    common_config=config,
    solver="Euler",
    seed=42,
)

X, A, info = sampler.sample(
    initial_features=[[0.1]],
    initial_adjacency=[[0.1]],
    terminal_time=1.0,
)

assert info["total_steps"][0] > 0
print("Smoke test passed")
```

## Monitoring

Key metrics to monitor in production:

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `total_steps` | Number of adaptive steps taken | 10-1000 (varies by problem) |
| `final_time` | Actual terminal time reached | Close to requested `terminal_time` |
| `dvs_history` | DVS values at each step | Should converge toward zero |
| `dt_history` | Timestep sizes | Between `dt_min` and `dt_max` |

## Troubleshooting

### "Maximum iterations exceeded"

Increase `terminal_time` or adjust `dt_max` to allow larger steps.

### Numerical instability (NaN/Inf in output)

- Ensure `eps_num` is not too small
- Check that initial noise values are reasonable
- Verify drift function output is bounded

### Slow performance

- The sampler is pure Python. For large graphs, consider the NumPy vectorization extensions in `docs/EXTENSIONS.md`.
- Profile with `python -m cProfile your_script.py` to identify bottlenecks.

## Version Pinning

For reproducible deployments, pin the exact version:

```
igasgd==0.1.0
```

Or in `pyproject.toml`:

```toml
dependencies = [
    "igasgd==0.1.0",
]
```
