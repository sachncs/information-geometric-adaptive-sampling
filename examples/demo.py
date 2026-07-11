"""Runnable demo of the DVS sampler with synthetic and model-approximation graph data.

This script provides two ways to drive the sampler:

* **Synthetic damping drift** (the default) -- a hand-written
  ``f(X, A, t) = -0.5 * X`` style drift.  Useful for smoke-testing the
  sampler pipeline without any neural network.
* **Simplified denoiser approximation** (``--use-approximation``) --
  wraps :class:`~igasgd.models.GruMApproximation` or
  :class:`~igasgd.models.GDSSApproximation` as the drift function.
  These are *not* the real GruM/GDSS networks (see the module docstring
  of :mod:`igasgd.models` for details) but they satisfy the same
  interface and let us demonstrate the sampler end-to-end.

Usage examples::

    # Euler solver with the GruM approximation
    python examples/demo.py --model GruM --dataset QM9 --solver Euler --use-approximation

    # Heun solver with the GDSS approximation
    python examples/demo.py --model GDSS --dataset QM9 --solver Heun --use-approximation

    # Reproducible run with a custom seed
    python examples/demo.py --model GruM --dataset QM9 --solver Euler --use-approximation --seed 42
"""

import argparse
import random
import statistics
import sys
from pathlib import Path

# Ensure the source tree is on the path when running directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from igasgd import (
    DATASET_CONFIGS,
    CommonConfig,
    DVSSampler,
    GDSSApproximation,
    GruMApproximation,
    LinearSchedule,
    decode_adjacency,
    get_dataset_config,
    make_drift_function,
)


def synthetic_drift(
    features: list[list[float]],
    adjacency: list[list[float]],
    time: float,
) -> tuple[list[list[float]], list[list[float]]]:
    """Simple synthetic drift that pushes values toward zero.

    Implements ``f_X = -damping * X`` and ``f_A = -damping * A``.  The
    time argument is ignored so the drift is autonomous.  This is
    useful for smoke-testing the sampler without instantiating a
    neural network.

    Args:
        features: Node feature matrix of shape ``(N, D)``.
        adjacency: Adjacency matrix of shape ``(N, N)``.
        time: Current diffusion time (ignored).

    Returns:
        A pair ``(f_X, f_A)`` of the same shapes as the inputs.
    """
    damping = 0.5
    f_x = [[-damping * x for x in row] for row in features]
    f_a = [[-damping * a for a in row] for row in adjacency]
    return f_x, f_a


def make_random_graph(num_nodes: int, feat_dim: int, rng: random.Random):
    """Generate a random initial graph state.

    Both the node features and adjacency entries are drawn uniformly
    from ``[0, 1)``.  The same RNG is used for both matrices so the
    randomness is reproducible.

    Args:
        num_nodes: Number of nodes in the graph.
        feat_dim: Feature dimensionality.
        rng: Random number generator.

    Returns:
        A pair ``(features, adjacency)`` of nested lists with shapes
        ``(num_nodes, feat_dim)`` and ``(num_nodes, num_nodes)``
        respectively.
    """
    features = [[rng.random() for _ in range(feat_dim)] for _ in range(num_nodes)]
    adjacency = [[rng.random() for _ in range(num_nodes)] for _ in range(num_nodes)]
    return features, adjacency


def run_demo(
    model: str = "GruM",
    dataset: str = "QM9",
    solver: str = "Euler",
    seed: int = 42,
    use_approximation: bool = False,
) -> None:
    """Run a sampling demo and print statistics.

    Selects the drift function (synthetic or model approximation),
    constructs a :class:`~igasgd.sampler.DVSSampler`, runs it on a
    random 5-node / 3-feature graph, and prints summary statistics
    including the number of steps, the mean / min / max / stddev of
    the adapted timestep, and the number of decoded edges.

    Args:
        model: Backbone model name (``"GruM"`` or ``"GDSS"``).  Only
            used when ``use_approximation=True``.
        dataset: Dataset name (e.g. ``"QM9"``, ``"Planar"``).
        solver: Solver name -- ``"Euler"`` or ``"Heun"``.
        seed: Random seed for reproducibility.
        use_approximation: If ``True``, instantiate a simplified
            denoising approximation instead of the synthetic drift.
    """
    common = CommonConfig()
    dataset_cfg = get_dataset_config(model, dataset)
    schedule = LinearSchedule(sigma_min=0.01, sigma_max=0.5)

    rng = random.Random(seed)
    features_0, adjacency_0 = make_random_graph(num_nodes=5, feat_dim=3, rng=rng)

    if use_approximation:
        if model == "GruM":
            approx = GruMApproximation(num_nodes=5, feature_dim=3, seed=seed)
        elif model == "GDSS":
            approx = GDSSApproximation(num_nodes=5, feature_dim=3, seed=seed)
        else:
            raise ValueError(f"No approximation available for model {model!r}")
        drift = make_drift_function(approx)
        print(f"Using {model} simplified approximation")
    else:
        drift = synthetic_drift
        print("Using synthetic damping drift")

    sampler = DVSSampler(
        drift_function=drift,
        noise_schedule=schedule,
        common_config=common,
        dataset_config=dataset_cfg,
        solver=solver,
        seed=seed,
    )

    print(f"Demo: model={model} dataset={dataset} solver={solver}")
    print(f"Common config: alpha={common.alpha} beta={common.beta} dt_base={common.dt_base}")
    print(
        f"Dataset config: kappa_ref={dataset_cfg.kappa_ref} "
        f"gamma={sampler.gamma} active_range={dataset_cfg.active_range}"
    )
    print(f"Initial graph: {len(features_0)} nodes, feature dim={len(features_0[0])}")
    print("-" * 60)

    features_t, adjacency_t, info = sampler.sample(
        initial_features=features_0,
        initial_adjacency=adjacency_0,
        terminal_time=1.0,
        verbose=True,
    )

    print("-" * 60)
    total_steps = int(info["total_steps"][0])
    final_time = info["final_time"][0]
    print(f"Sampling complete in {total_steps} steps (final time={final_time:.6f})")

    dts = info["dt"]
    print(f"Mean dt: {statistics.mean(dts):.6f}")
    print(f"Min dt: {min(dts):.6f}  Max dt: {max(dts):.6f}")

    adjacency_bin = decode_adjacency(adjacency_t, threshold=0.5)
    edge_count = sum(sum(row) for row in adjacency_bin)
    print(f"Decoded adjacency edges (threshold=0.5): {int(edge_count)}")

    if dts:
        print(f"dt stddev: {statistics.stdev(dts):.6f}")


def main() -> None:
    """CLI entry point.

    Parses the command-line arguments and delegates to
    :func:`run_demo`.  A missing ``(model, dataset)`` configuration is
    caught and reported as a friendly error message before exiting
    with status ``1``.
    """
    parser = argparse.ArgumentParser(
        description="DVS sampler demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="GruM",
        choices=["GruM", "GDSS"],
        help="Backbone model name",
    )
    parser.add_argument(
        "--dataset",
        default="QM9",
        help="Dataset name",
    )
    parser.add_argument(
        "--solver",
        default="Euler",
        choices=["Euler", "Heun"],
        help="SDE solver",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use-approximation",
        action="store_true",
        help="Use a simplified model approximation instead of synthetic drift",
    )
    args = parser.parse_args()

    try:
        run_demo(args.model, args.dataset, args.solver, args.seed, args.use_approximation)
    except KeyError as exc:
        print(f"Error: {exc}")
        print(f"Available configs: {list(DATASET_CONFIGS.keys())}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
