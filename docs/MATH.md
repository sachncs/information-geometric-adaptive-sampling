# Mathematical Background

This document restates the core mathematics behind the DVS-driven adaptive sampler and maps every equation to its implementation.  The notation follows arXiv:2605.00250v1 exactly.

---

## 1. Preliminaries

### Reverse-Time SDE (Equation 1)

The generative process is the time-reversal of a forward diffusion:

```
d x_t = f_t dt + g_t d w_t
```

where:
- `f_t = f^{psi}(x_t, t)` is the drift predicted by a trained denoising network.
- `g_t` is the noise schedule (locally constant during each step).
- `w_t` is a standard Wiener process.

For graph data the state splits into two modalities:
- `X_t in R^{N x D}` — node features
- `A_t in R^{N x N}` — adjacency / edge features

Each modality has its own drift:

```
d X_t = f_X(X_t, A_t, t) dt + g_t d w_t^{(X)}
d A_t = f_A(X_t, A_t, t) dt + g_t d w_t^{(A)}
```

### Euler-Maruyama Discretisation (Equation 2)

The simplest first-order discretisation is:

```
x_{t+dt} = x_t + f_t dt + g_t sqrt(dt) z_t,   z_t ~ N(0, I)
```

This is implemented in `euler_step()` (`sampler.py`).

---

## 2. Information Geometry on the Transition Manifold

The paper introduces a statistical manifold whose points are transition kernels `p(x_{t+dt} | x_t; f_t)`.  The Fisher-Rao metric on this manifold yields a line element (Eq. 3--12 in the paper):

```
ds^2 = ||f_k - f_{k-1}||_2^2 / g_t^2
```

This geometric distance motivates the **Drift Variation Score** as a proxy for local curvature.

---

## 3. Drift Variation Score (DVS)

### Equation 13

For each modality we define:

```
V_{X,k} = ||f_{X,k} - f_{X,k-1}||_2^2 / (g_{t_k}^2 + epsilon_num)
V_{A,k} = ||f_{A,k} - f_{A,k-1}||_2^2 / (g_{t_k}^2 + epsilon_num)
```

where `epsilon_num = 1e-12` prevents division-by-zero in the low-noise regime.

**Implementation:** `compute_drift_variation_score()` (`sampler.py`).

**Validation:**
- If `f_k == f_{k-1}` then `V = 0`.
- If `g_t = 0` then `V = ||diff||^2 / epsilon_num` (very large, which correctly forces a tiny timestep).
- The function validates that the two drift matrices have identical shapes and raises `ValueError` otherwise.

---

## 4. EMA Smoothing

### Equation 14

Raw DVS values are noisy, so an exponential moving average is applied:

```
Vbar_X <- alpha * V_{X,k} + (1 - alpha) * Vbar_X
Vbar_A <- alpha * V_{A,k} + (1 - alpha) * Vbar_A
```

with `alpha = 0.2`.

**Implementation:** `update_ema()` (`sampler.py`).

**Properties:**
- `alpha = 1.0` gives no smoothing (Vbar = V_k).
- `alpha = 0.0` freezes the estimate (Vbar never changes).
- The EMA is initialised to `0.0` at `t = 0`.

---

## 5. Power-Law Step-Size Scaling

### Equation 15

The smoothed DVS drives the timestep via a power law:

```
dt_X = clip(dt_base * (kappa_ref / Vbar_X)^beta, dt_min, dt_max)
dt_A = clip(dt_base * (kappa_ref / Vbar_A)^beta, dt_min, dt_max)
```

with `beta = 0.5` (square-root scaling).

**Implementation:** `compute_timestep()` (`sampler.py`).

**Physical interpretation:**
- When curvature `Vbar` is large relative to `kappa_ref`, the ratio `< 1` and `dt` shrinks to resolve fine structure.
- When curvature is small, the ratio `> 1` and `dt` grows to `dt_max` to save compute.
- `kappa_ref` acts as a set-point: at `Vbar = kappa_ref` we recover the base timestep `dt_base`.

**Edge cases handled in code:**
- `Vbar = 0` -> denominator replaced by `epsilon_num`, yielding `dt = dt_max` (fast motion where no drift variation is detected).
- `Vbar -> inf` -> `dt` clips to `dt_min`.
- `beta = 2.0` would make the response quadratic; the unit tests verify clipping in that regime.

---

## 6. Bottleneck Principle

To keep the two modalities synchronised in time, the sampler takes the more conservative (smaller) timestep:

```
dt_k = min(dt_X, dt_A)
```

**Implementation:** `DVSSampler.sample()` (`sampler.py`).

This ensures that neither X nor A races ahead, preserving the coupled dynamics.

---

## 7. Global Variation Refresh

After each adaptive step, the two EMA states are replaced by a single aggregated value:

```
Vbar_X, Vbar_A <- gamma * (Vbar_X + Vbar_A)
```

**Implementation:** `global_refresh()` (`sampler.py`).

**Why this works:**
- Both modalities receive the same combined score, so future bottleneck decisions are not biased by whichever modality happened to be smoother.
- `gamma` is dataset- and solver-specific (see Table 7).
- If `gamma = 0`, both scores are reset to `0.0` (adaptive behaviour is disabled until new drift variation is observed).

---

## 8. Active Ranges

The paper observes that DVS is most informative only during certain phases of the trajectory (e.g., early and late times for GDSS/QM9).  The sampler therefore supports *active ranges*: time intervals inside which DVS is computed, and outside which `dt_k = dt_base`.

**Notation:**
- `[(0.0, 1.0)]` — DVS is always active.
- `[(0.0, 0.2), (0.95, 1.0)]` — DVS active only in the union of these intervals.
- `[]` (empty) — equivalent to `[(0.0, 1.0)]` (always active).

**Implementation:** `DatasetConfig.is_active()` and `in_active_range()` (`utils.py`).

---

## 9. Heun Predictor-Corrector (Algorithm 3)

The Heun solver is a second-order method:

1. **Predictor** (Euler):
   ```
   X_hat = X + f^{(1)} dt + g sqrt(dt) Z
   ```

2. **Corrector drift** at `t + dt`:
   ```
   f^{(2)} = drift_function(X_hat, A_hat, t + dt)
   ```

3. **Corrector** (trapezoidal average):
   ```
   X_k = X + 0.5 * (f^{(1)} + f^{(2)}) dt + g sqrt(dt) Z
   ```

The same noise realisation `Z` is used in both stages so that the corrector remains consistent with the same Brownian path.

**Implementation:** `heun_step()` (`sampler.py`).

**Accuracy:** On the test problem `dx/dt = -x` (analytical solution `exp(-t)`), Heun is measurably closer to the true solution than Euler for the same step count.

---

## 10. Numerical Constants (Table 6)

| Symbol | Value | Role |
|--------|-------|------|
| alpha  | 0.2   | EMA smoothing coefficient |
| beta   | 0.5   | Power-law exponent |
| dt_base| 1e-3  | Base timestep |
| dt_min | 2e-4  | Lower bound on dt |
| dt_max | 5e-3  | Upper bound on dt |
| eps_num| 1e-12 | Division-by-zero guard |
| eps_bound| 1e-6| Boundary tolerance (loop termination) |

**Implementation:** `CommonConfig` dataclass (`config.py`).

---

## 11. Dataset-Specific Hyperparameters (Table 7)

Each dataset/model pair has a reference curvature `kappa_ref`, an aggregation factor `gamma` (per solver), and optional active ranges.

| Model | Dataset | kappa_ref | Euler gamma | Heun gamma | Active Range |
|-------|---------|-----------|-------------|------------|--------------|
| GruM  | QM9     | 1.0       | 0.22        | 0.23       | [0, 1] |
| GruM  | ZINC250k| 5.0       | 0.02        | 0.04       | [0, 1] |
| GruM  | Planar  | 10.0      | 0.31        | 0.30       | [0.5, 1.0] |
| GruM  | SBM     | 10.0      | 0.26        | 0.26       | [0.4, 1.0] |
| GDSS  | QM9     | 1.0       | 0.68        | --         | [0,0.2] U [0.95,1] |
| GDSS  | Ego-small| 0.2      | --          | --         | [0, 1] |
| GDSS  | Grid    | 0.1       | --          | --         | [0, 1] |
| GDSS  | Community-small | 0.1 | --       | --         | [0, 1] |

**Implementation:** `DATASET_CONFIGS` dictionary and `DatasetConfig` dataclass (`config.py`).

---

## 12. Noise Schedule `g(t)`

The paper uses `g_t` but does not specify its exact functional form.  We provide three common parameterisations:

**Linear:**
```
g(t) = sigma_max * t + sigma_min * (1 - t)
```

**Cosine (approximate):**
```
g(t) = tan((t + offset) / (1 + offset) * pi / 2)
```

**Polynomial:**
```
g(t) = sigma_max * t^exponent + sigma_min
```

All schedules are callables `g(t: float) -> float` and can be swapped by the user.

**Implementation:** `LinearSchedule`, `CosineSchedule`, `PolynomialSchedule` (`schedule.py`).

---

## 13. Graph Decoding

After sampling, the continuous adjacency matrix must be decoded to discrete edges.  The paper mentions this step but does not specify the threshold or activation.  We provide two options:

**Threshold decoder:**
```
A_bin[i,j] = 1.0  if A_cont[i,j] >= threshold else 0.0
```

**Sigmoid decoder:**
```
A_bin[i,j] = 1.0  if sigmoid(A_cont[i,j]) >= threshold else 0.0
```

Default threshold is `0.5`.

**Implementation:** `decode_adjacency()` and `sigmoid_decode_adjacency()` (`utils.py`).
