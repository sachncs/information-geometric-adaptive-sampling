# Optional Extensions

These are enhancements that improve efficiency, readability, numerical stability, and modularity. They are **clearly separated from the baseline** and do not affect the core paper reproduction.

Each extension is self-contained and can be adopted independently.

---

## Table of Contents

- [1. Numerical Stability](#1-numerical-stability)
- [2. Efficiency](#2-efficiency)
- [3. Readability / Instrumentation](#3-readability--instrumentation)
- [4. Modularity](#4-modularity)
- [5. Validation Helpers](#5-validation-helpers)
- [6. How to Apply](#6-how-to-apply)

---

## 1. Numerical Stability

### Adaptive epsilon for DVS denominator

In very low-noise regimes, `g_t` can become numerically zero. The baseline already adds `eps_num = 1e-12`. An extension could adapt `eps_num` to the floating-point precision of the runtime environment:

```python
import sys
eps_num = float(sys.float_info.epsilon)  # ~2.2e-16 on IEEE-754 double
```

Or use `math.ulp(1.0)` for the unit in the last place.

### Soft clipping

Instead of hard `min/max` clipping in Eq. 15, a smooth sigmoid clip could reduce gradient discontinuities if the sampler were ever differentiated:

```python
def soft_clip(value, lower, upper, steepness=100.0):
    """Smooth approximation of clip via sigmoid blending."""
    return lower + (upper - lower) * (1.0 / (1.0 + math.exp(-steepness * (value - (lower + upper) / 2))))
```

Not needed for inference-only sampling.

---

## 2. Efficiency

### Vectorisation with NumPy

The baseline uses pure Python lists for portability. A drop-in NumPy backend would:
- Replace `list[list[float]]` with `np.ndarray`.
- Use `np.linalg.norm` for DVS.
- Use broadcasting for Euler/Heun steps.
- Speed up large graphs (N > 100) by 10-100x.

**Impact:** Trivial for the numerical kernels; the adaptive logic remains identical.

**Files to touch:** `sampler.py` (all list operations), `models.py` (matrix multiplication), `utils.py` (decoding).

### Cached drift reuse for Heun

Algorithm 3 already caches the first-stage drift `f^{(1)}` for DVS. An extension could also cache the second-stage drift `f^{(2)}` across the boundary clip, though the paper does not describe this.

### JIT compilation

If NumPy is adopted, Numba `@njit` on `compute_drift_variation_score`, `euler_step`, and `heun_step` would further reduce overhead:

```python
from numba import njit

@njit
def compute_drift_variation_score(...):
    ...
```

---

## 3. Readability / Instrumentation

### Rich history logging

The baseline returns `dt`, `time`, `v_x`, `v_a`, `smoothed_x`, `smoothed_a`. An extension could also log:
- `f_norm_x`, `f_norm_a` (drift magnitudes)
- `g_t` at each step
- Effective information distance per step (Eq. 12)

**Implementation:** Add keys to the `info` dict in `DVSSampler.sample()`.

### Plotting utilities

A `plot_trajectory(info)` helper could visualise:
- `dt` vs `time`
- `smoothed_x` and `smoothed_a` vs `time`
- Overlay active ranges

**Implementation:** Add a new module `plots.py` (requires `matplotlib` as an optional dependency).

---

## 4. Modularity

### Pluggable solvers

The baseline supports Euler and Heun via an `if solver == ...` branch. A more modular design would register solvers in a dictionary:

```python
SOLVERS = {
    "Euler": euler_step,
    "Heun": heun_step,
    "DPM": dpm_solver_step,
}
```

**Implementation:** Replace the `if/else` block in `DVSSampler.sample()` with:

```python
solver_fn = SOLVERS[self._solver]
next_features, next_adjacency = solver_fn(...)
```

### Configurable schedulers

Add `CosineSchedule`, `PolynomialSchedule`, and `LearnedSchedule` wrappers so users can experiment with `g(t)` without modifying `sampler.py`.

The baseline already supports this: any `Callable[[float], float]` can be passed as `noise_schedule`. The extension is to provide a registry or factory:

```python
SCHEDULES = {
    "linear": LinearSchedule,
    "cosine": CosineSchedule,
    "polynomial": PolynomialSchedule,
}
```

---

## 5. Validation Helpers

### Synthetic drift suites

For unit testing, add more complex synthetic drifts (oscillatory, piecewise) to stress-test the adaptive step-size logic:

```python
def oscillatory_drift(x, a, t):
    omega = 2.0 * math.pi
    return [[math.sin(omega * t) * v for v in row] for row in x], ...
```

### Fidelity regression tests

Serialise the `info` dict from a known-good run and assert future runs match within tolerance. This catches unintentional algorithmic drift.

**Implementation:** Add a `tests/test_regression.py` file that loads a golden JSON and compares key statistics (mean dt, total steps, final time).

---

## 6. How to Apply

Each extension is independent. For example, to adopt the NumPy backend:

```python
# Replace in sampler.py and tests
import numpy as np

# State becomes np.ndarray
# compute_drift_variation_score becomes:
def compute_drift_variation_score(f_curr_x, f_prev_x, f_curr_a, f_prev_a, g_t, eps_num):
    diff_x = np.array(f_curr_x) - np.array(f_prev_x)
    diff_a = np.array(f_curr_a) - np.array(f_prev_a)
    denom = g_t**2 + eps_num
    v_x = float(np.sum(diff_x**2) / denom)
    v_a = float(np.sum(diff_a**2) / denom)
    return v_x, v_a
```

No other logic changes.

---

## Compatibility Notes

- All extensions are additive; removing them should restore the baseline exactly.
- The baseline avoids external dependencies so that it runs on any Python 3.10+ environment. Extensions may relax this constraint.
- Before adding an extension, ensure the existing test suite still passes.
