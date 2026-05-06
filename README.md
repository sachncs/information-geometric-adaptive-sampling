# igasgd — Information-Geometric Adaptive Sampling for Graph Diffusion

Pure Python reproduction of the DVS-driven adaptive sampler from:

> **Information-Geometric Adaptive Sampling for Graph Diffusion**  
> arXiv:2605.00250v1

## What this reproduces

This package faithfully implements the training-free DVS sampler that can be
plugged into existing graph diffusion models.  It includes:

- **Algorithm 1:** DVS-driven adaptive sampler (meta-algorithm)
- **Algorithm 2:** DVS-Euler--Maruyama adaptive sampler
- **Algorithm 3:** DVS-Heun adaptive sampler
- **Table 6:** Common hyperparameters (alpha, beta, dt bounds, epsilons)
- **Table 7:** Dataset-specific hyperparameters for GruM and GDSS
- **Simplified model approximations:** Educational stand-in denoising networks
  for GruM and GDSS that satisfy the drift interface (clearly marked as
  approximations)

## Quick start

```bash
# Run the expanded test suite
python tests/test_sampler.py

# Run demos
python examples/demo.py --model GruM --dataset QM9 --solver Euler
python examples/demo.py --model GDSS  --dataset QM9 --solver Euler

# Use a model approximation instead of synthetic drift
python examples/demo.py --model GruM --dataset QM9 --solver Euler --use-approximation
```

## Usage

```python
from igasgd import (
    CommonConfig,
    get_dataset_config,
    DVSSampler,
    LinearSchedule,
    GruMApproximation,
    make_drift_function,
)

# 1. Load or define your trained denoising network
#    (here we use the simplified approximation as a stand-in)
approx = GruMApproximation(num_nodes=9, feature_dim=4, seed=42)
drift = make_drift_function(approx)

# 2. Pick configuration from the paper
common = CommonConfig()
dataset = get_dataset_config("GruM", "QM9")

# 3. Build sampler
sampler = DVSSampler(
    drift_function=drift,
    noise_schedule=LinearSchedule(sigma_min=0.01, sigma_max=0.5),
    common_config=common,
    dataset_config=dataset,
    solver="Euler",  # or "Heun"
    seed=42,
)

# 4. Sample a graph
features_0 = [[...], ...]  # initial noise for node features
adjacency_0 = [[...], ...]  # initial noise for adjacency
features_t, adjacency_t, info = sampler.sample(
    initial_features=features_0,
    initial_adjacency=adjacency_0,
    terminal_time=1.0,
)

# info contains step count, dt history, DVS history, etc.
print(f"Steps: {info['total_steps'][0]}, Final time: {info['final_time'][0]}")
```

## Hyperparameters

Common (Table 6):

| Parameter | Symbol | Value |
|-----------|--------|-------|
| EMA Smoothing | alpha | 0.2 |
| Sensitivity Exponent | beta | 0.5 |
| Base Timestep | dt_base | 1e-3 |
| Minimum Timestep | dt_min | 2e-4 |
| Maximum Timestep | dt_max | 5e-3 |
| Stability Constant | eps_num | 1e-12 |
| Boundary Tolerance | eps_bound | 1e-6 |

Dataset-specific (Table 7) include `kappa_ref`, `gamma` per solver, and optional
`active_range`.

## Project layout

```
igasgd/
├── src/igasgd/
│   ├── __init__.py
│   ├── config.py        # Table 6 & 7 hyperparameters
│   ├── sampler.py       # Algorithms 1, 2, 3
│   ├── models.py        # Simplified GruM / GDSS approximations
│   ├── schedule.py      # Noise schedule utilities
│   ├── utils.py         # Clipping, active ranges, decoding
├── tests/
│   └── test_sampler.py  # 38 unit tests
├── examples/
│   └── demo.py          # Runnable demo with CLI
├── docs/
│   └── EXTENSIONS.md    # Optional enhancements
├── REPRODUCTION_SUMMARY.md
├── FIDELITY_REPORT.md
└── README.md
```

## Known gaps

1. **Denoising networks:** The actual GruM/GDSS architectures and pretrained
   weights are not publicly released.  We provide simplified approximations
   (`GruMApproximation`, `GDSSApproximation`) that satisfy the drift interface
   but do **not** perform real message passing or graph attention.  All
   invented architectural details are clearly documented.
2. **Noise schedule `g(t)`:** The exact functional form used in the paper is
   not explicitly stated.  We provide common parameterizations (linear, cosine,
   polynomial) and allow users to supply their own callable.
3. **Evaluation metrics (FCD, NSPDK):** Require RDKit and graph-kernel
   libraries.  These are outside the scope of a pure Python reproduction and
   are not included.
4. **Training code:** Out of scope; this is a sampler-only reproduction.

## License

MIT — Provided for research and educational purposes.  See the original paper
for authorship and citation.
