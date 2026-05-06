# Usage Guide

This guide shows how to use the `igasgd` package from a clean start, progressing from a minimal example to advanced integration patterns.

---

## 1. Installation

No external dependencies are required.  The package is pure Python and works with Python 3.10 or newer.

```bash
# From the repository root
cd /path/to/igasgd
python -m pip install -e .
```

To install development tools (pytest, mypy, ruff):

```bash
python -m pip install -e ".[dev]"
```

---

## 2. Minimal Example

```python
from igasgd import (
    CommonConfig,
    DatasetConfig,
    DVSSampler,
    constant_schedule,
)


def my_drift(features, adjacency, time):
    """Toy drift that pushes everything toward zero."""
    f_x = [[-0.5 * v for v in row] for row in features]
    f_a = [[-0.5 * v for v in row] for row in adjacency]
    return f_x, f_a


# 1. Configuration
common = CommonConfig()
dataset = DatasetConfig(
    model="Test", dataset="Test",
    kappa_ref=1.0,
    gamma_euler=0.5, gamma_heun=0.5,
    active_range=[(0.0, 1.0)],
)

# 2. Sampler
sampler = DVSSampler(
    drift_function=my_drift,
    noise_schedule=constant_schedule(0.1),
    common_config=common,
    dataset_config=dataset,
    solver="Euler",
    seed=42,
)

# 3. Initial noise
x0 = [[1.0, 0.0], [0.0, 1.0]]
a0 = [[0.5, 0.3], [0.3, 0.5]]

# 4. Sample
x_t, a_t, info = sampler.sample(x0, a0, terminal_time=1.0, verbose=False)

print(f"Steps taken: {int(info['total_steps'][0])}")
print(f"Final time:  {info['final_time'][0]:.6f}")
```

---

## 3. Using Paper Configurations

Instead of hand-writing `DatasetConfig`, load the official Table 7 values:

```python
from igasgd import CommonConfig, get_dataset_config, DVSSampler

common = CommonConfig()
dataset = get_dataset_config("GruM", "QM9")

sampler = DVSSampler(
    drift_function=my_drift,
    noise_schedule=...,  # your schedule
    common_config=common,
    dataset_config=dataset,
    solver="Euler",
    seed=42,
)
```

Available configurations:

| Model | Dataset |
|-------|---------|
| GruM  | QM9, ZINC250k, Planar, SBM |
| GDSS  | QM9, Ego-small, Grid, Community-small |

---

## 4. Switching Solvers

Change a single argument:

```python
sampler_euler = DVSSampler(..., solver="Euler")
sampler_heun  = DVSSampler(..., solver="Heun")
```

**When to use which:**
- **Euler** — faster, first-order, good for exploratory runs.
- **Heun** — more accurate, second-order, requires an extra drift evaluation per step (roughly 2x drift cost).

Both solvers share the same DVS-driven adaptive timestep logic.

---

## 5. Using the Simplified Model Approximations

For smoke testing without a real denoising network:

```python
from igasgd import GruMApproximation, make_drift_function

approx = GruMApproximation(num_nodes=9, feature_dim=4, seed=42)
drift = make_drift_function(approx)

sampler = DVSSampler(
    drift_function=drift,
    noise_schedule=...,
    common_config=common,
    dataset_config=dataset,
    solver="Heun",
    seed=42,
)
```

The approximations are deterministic MLPs.  They do **not** perform message passing or graph attention, but they satisfy the drift interface and produce non-trivial trajectories.

---

## 6. Custom Noise Schedules

Any callable `g(t) -> float` works:

```python
from igasgd import LinearSchedule, CosineSchedule, PolynomialSchedule

# Linear interpolation
sched = LinearSchedule(sigma_min=0.01, sigma_max=0.5)

# Cosine
sched = CosineSchedule(offset=0.008)

# Polynomial (e.g., square-root-like)
sched = PolynomialSchedule(exponent=0.5, sigma_min=0.01, sigma_max=1.0)

# Custom lambda
sched = lambda t: 0.1 + 0.9 * t ** 2
```

---

## 7. Decoding the Output Graph

The sampler returns a continuous adjacency matrix.  To decode it to discrete edges:

```python
from igasgd import decode_adjacency

adjacency_bin = decode_adjacency(a_t, threshold=0.5)
edge_count = sum(sum(row) for row in adjacency_bin)
print(f"Edges decoded: {edge_count}")
```

For a smooth sigmoid decoder:

```python
from igasgd import sigmoid_decode_adjacency

adjacency_bin = sigmoid_decode_adjacency(a_t, threshold=0.5)
```

---

## 8. Inspecting the Sampling History

The `info` dict returned by `sample()` contains rich diagnostics:

```python
import statistics

info = sampler.sample(x0, a0, terminal_time=1.0, verbose=False)[2]

dts = info["dt"]
print(f"Mean dt: {statistics.mean(dts):.6f}")
print(f"Min dt:  {min(dts):.6f}")
print(f"Max dt:  {max(dts):.6f}")

# Identify where DVS was most active
max_vx = max(info["v_x"])
max_idx = info["v_x"].index(max_vx)
print(f"Peak DVS at step {int(info['steps'][max_idx])}, time={info['time'][max_idx]:.4f}")
```

---

## 9. Reproducibility

Fix `seed` in `DVSSampler`:

```python
sampler1 = DVSSampler(..., seed=123)
sampler2 = DVSSampler(..., seed=123)

x1, a1, i1 = sampler1.sample(x0, a0, terminal_time=0.1)
x2, a2, i2 = sampler2.sample(x0, a0, terminal_time=0.1)

assert i1["dt"] == i2["dt"]
assert x1 == x2
```

Note: this only guarantees reproducibility if `drift_function` itself is deterministic.

---

## 10. CLI Demo

A runnable demo script is provided in `examples/demo.py`:

```bash
# Synthetic drift
python examples/demo.py --model GruM --dataset QM9 --solver Euler

# With simplified model approximation
python examples/demo.py --model GruM --dataset QM9 --solver Euler --use-approximation

# Different dataset
python examples/demo.py --model GDSS --dataset Grid --solver Heun --seed 99
```

The demo prints per-step diagnostics and a summary statistics block at the end.

---

## 11. Integration into Your Own Project

The typical integration pattern is:

1. **Load your real denoising network** (PyTorch, JAX, etc.) and wrap it in a pure-Python callable:
   ```python
   def my_drift(features, adjacency, time):
       # Convert lists to tensors, run model, convert back to lists
       return drift_x, drift_a
   ```

2. **Pick or define a noise schedule** `g(t)` that matches your diffusion setup.

3. **Instantiate `DVSSampler`** with your drift, schedule, and the paper configs.

4. **Call `sample()`** and decode the result.

5. **Replace `decode_adjacency()`** with your own post-processing if you have domain-specific decoding rules.

The sampler is entirely stateless between calls (except for the RNG), so you can reuse the same `DVSSampler` instance for multiple graphs.
