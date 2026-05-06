# Architecture and Data Flow

This document explains how the `igasgd` package is structured, what each component does, and how data flows through a sampling run.

---

## 1. Design Goals

1. **Training-free plug-in:** The sampler does not train anything; it wraps an existing denoising network and its noise schedule.
2. **Pure Python:** No NumPy, PyTorch, or external heavy dependencies. Everything runs with the standard library.
3. **Paper-faithful:** Every equation, algorithm, and table from the paper has a direct code counterpart.
4. **Modular:** Solvers, schedules, and configurations are swappable without touching the core sampler loop.

---

## 2. File-Level Responsibilities

```
src/igasgd/
  __init__.py      — Public API exports.
  config.py        — Hyperparameter dataclasses (Tables 6 & 7).
  sampler.py       — Core DVS sampler and solver steps (Algorithms 1–3).
  models.py        — Simplified denoising network approximations.
  schedule.py      — Noise schedule callables (linear, cosine, polynomial).
  utils.py         — Clipping, active-range checks, graph decoding.
```

### 2.1 `config.py`

Contains two frozen dataclasses:

- `CommonConfig` — Table 6 constants shared across all experiments.
- `DatasetConfig` — Table 7 constants (model, dataset, `kappa_ref`, `gamma_euler`, `gamma_heun`, `active_range`).

`DatasetConfig.is_active(time)` is the canonical predicate for deciding whether DVS should be computed at a given diffusion time.

`DATASET_CONFIGS` is a global dictionary keyed by `(model, dataset)` tuples. `get_dataset_config()` performs lookup and raises `KeyError` with a helpful message on misses.

### 2.2 `sampler.py`

The heart of the package. It contains:

- **Low-level equation helpers:**
  - `_squared_l2_difference()` — element-wise squared L2 norm over nested lists.
  - `compute_drift_variation_score()` — Equation 13.
  - `update_ema()` — Equation 14.
  - `compute_timestep()` — Equation 15.
  - `global_refresh()` — global variation refresh.

- **Solver steps:**
  - `euler_step()` — Equation 2 / Algorithm 2.
  - `heun_step()` — Algorithm 3.

- **Orchestrator:**
  - `DVSSampler` — Implements the full adaptive loop (Algorithm 1 meta-algorithm).

#### The `DVSSampler` Loop

```
Input: X_0, A_0, T, drift_function, noise_schedule, config, solver
Initialise: t = 0, k = 1, Vbar_X = 0, Vbar_A = 0
Cache: previous_drift = None

While t < T - eps_bound:
    1. Evaluate drift at current state:
       f_X, f_A = drift_function(X_prev, A_prev, t)
       g_t = noise_schedule(t)

    2. Decide whether DVS is active:
       active = (k > 1) and (t in active_range) and (previous_drift exists)

    3. If active:
          a. Compute DVS via Equation 13 using cached previous drift.
          b. Update EMA via Equation 14.
          c. Compute dt_X and dt_A via Equation 15.
          d. dt_k = min(dt_X, dt_A)   (Bottleneck principle)
          e. Global refresh: Vbar_X, Vbar_A = gamma * (Vbar_X + Vbar_A)
       Else:
          dt_k = dt_base

    4. Clip dt_k so we do not overshoot T:
       dt_k = min(dt_k, T - t)

    5. Apply solver step:
       noise_scale = g_t * sqrt(dt_k)
       if solver == "Euler":
           X_k, A_k = euler_step(...)
       else:
           X_k, A_k = heun_step(...)

    6. Record history and advance:
       append step info to info dict
       cache current drift as previous drift
       t += dt_k
       k += 1
       prev_features = X_k
       prev_adjacency = A_k

Output: X_T, A_T, info_dict
```

### 2.3 `models.py`

Provides simplified stand-in denoising networks that satisfy the `drift_function(X, A, t)` interface.

Because the real GruM/GDSS architectures and pretrained weights are not publicly released, this module contains:

- `SimpleGraphDenoiser` — base class with a 3-layer MLP, Xavier uniform initialisation, and sinusoidal time embedding.
- `GruMApproximation(hidden_dim=32)` — educational stand-in for GruM.
- `GDSSApproximation(hidden_dim=24)` — educational stand-in for GDSS.
- `make_drift_function(model)` — wraps any `SimpleGraphDenoiser` into the standard drift callable.

**Important:** These are not the real networks. They exist only for smoke tests, interface validation, and demos. When the real weights become available, replace `make_drift_function(your_real_model)`.

### 2.4 `schedule.py`

Defines `NoiseSchedule = Callable[[float], float]`.

Concrete implementations:
- `LinearSchedule`
- `CosineSchedule`
- `PolynomialSchedule`
- `constant_schedule(value)` — useful for deterministic testing.

All schedules are stateless callables and can be swapped by passing a different instance to `DVSSampler`.

### 2.5 `utils.py`

Small, stateless helpers:
- `clip_value()` — scalar clamp with NaN propagation.
- `in_active_range()` — interval membership (inclusive).
- `decode_adjacency()` — threshold binarizer.
- `sigmoid_decode_adjacency()` — sigmoid + threshold binarizer.

---

## 3. Data Flow Diagram

```
+---------------+     +----------------+     +------------------+
|  User code    |     |  config.py     |     |  schedule.py     |
|  (drift net)  |     |  CommonConfig  |     |  NoiseSchedule   |
+-------+-------+     +----------------+     +------------------+
        |                     |                       |
        | drift_function      | common_config         | noise_schedule
        v                     v                       v
+-----------------------------------------------------------+
|                        DVSSampler                          |
|  (sampler.py)                                              |
|                                                            |
|  while t < T:                                              |
|    drift = drift_function(X, A, t)                         |
|    if active:                                              |
|      v_x, v_a = compute_drift_variation_score(...)        |
|      s_x, s_a = update_ema(v_x, v_a, ...)                 |
|      dt_x  = compute_timestep(s_x, ...)                   |
|      dt_a  = compute_timestep(s_a, ...)                   |
|      dt    = min(dt_x, dt_a)                              |
|      s_x, s_a = global_refresh(s_x, s_a, gamma)          |
|    else:                                                   |
|      dt = dt_base                                          |
|    dt = clip(dt, 0, T - t)                                 |
|    X, A = euler_step / heun_step(...)                     |
|    record_history()                                        |
|    t += dt                                                 |
+-----------------------------------------------------------+
        |
        v
+-----------------------------------------------------------+
|  Output: X_T, A_T, info_dict                              |
|  Optional: decode_adjacency(A_T)                          |
+-----------------------------------------------------------+
```

---

## 4. Key Invariants

1. **Shape preservation:** Every solver step returns `(X_k, A_k)` with exactly the same shapes as `(X_{k-1}, A_{k-1})`.
2. **Time monotonicity:** `t` never decreases. `dt_k` is always non-negative because it is clipped to `[dt_min, dt_max]` and then to `[0, T - t]`.
3. **Determinism:** Fixing `seed` in `DVSSampler` makes the entire trajectory reproducible (drift evaluations are deterministic by assumption).
4. **Bottleneck synchrony:** Both modalities share the same `dt_k`, so `time` is identical for X and A at every step.
5. **EMA non-negativity:** If all drift differences are non-negative (squared norm), and `alpha in [0, 1]`, then `Vbar` stays non-negative.
6. **Active-range inclusivity:** A time exactly equal to a boundary is considered inside the interval.

---

## 5. Error Handling Strategy

The codebase raises explicit exceptions for contract violations rather than silently producing wrong results:

- `ValueError` from `DVSSampler.__init__` when:
  - `solver` is not `"Euler"` or `"Heun"`.
  - The requested solver has no `gamma` in `DatasetConfig`.
- `ValueError` from `_squared_l2_difference()` when drift matrices have mismatched shapes.
- `KeyError` from `get_dataset_config()` when the `(model, dataset)` pair is unknown.

No exceptions are raised for numerical edge cases such as `g_t = 0` or `Vbar = 0`; these are handled by the `epsilon_num` stabiliser and clipping.

---

## 6. Extensibility Points

If you want to extend the sampler, the cleanest insertion points are:

1. **New solver:** Add a solver step function in `sampler.py` and extend the `solver` branch in `DVSSampler.sample()`.
2. **New schedule:** Subclass a callable in `schedule.py` (or just pass any `Callable[[float], float]`).
3. **New model wrapper:** Provide any `drift_function(X, A, t) -> (f_X, f_A)` callable; it does not need to inherit from `SimpleGraphDenoiser`.
4. **New dataset config:** Add a row to `DATASET_CONFIGS` in `config.py`.
5. **New decoder:** Add a decoding function in `utils.py` and call it after `sampler.sample()`.

See [EXTENSIONS.md](EXTENSIONS.md) for detailed enhancement ideas.
