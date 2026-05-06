"""Hyperparameter configurations extracted from Tables 6 and 7.

Table 6 lists common hyperparameters kept constant across all datasets and models.
Table 7 lists dataset-specific hyperparameters including reference curvature,
aggregation factors, and active time intervals.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class CommonConfig:
    """Common hyperparameters for the DVS-driven adaptive sampler.

    These values were kept constant across all benchmarks in the paper
    and are relatively robust to setting changes.

    Attributes:
        alpha: EMA smoothing coefficient (default 0.2).
        beta: Sensitivity exponent controlling response to curvature (default 0.5).
        dt_base: Base timestep used as initial step size (default 1e-3).
        dt_min: Lower bound on step size to prevent numerical stalling (default 2e-4).
        dt_max: Upper bound on step size to maintain stability (default 5e-3).
        eps_num: Stability constant preventing division-by-zero (default 1e-12).
        eps_bound: Boundary tolerance ensuring robust termination at time T (default 1e-6).
    """

    alpha: float = 0.2
    beta: float = 0.5
    dt_base: float = 1e-3
    dt_min: float = 2e-4
    dt_max: float = 5e-3
    eps_num: float = 1e-12
    eps_bound: float = 1e-6


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-specific hyperparameters and adaptive intervals.

    Attributes:
        model: Name of the backbone model (e.g., "GruM", "GDSS").
        dataset: Name of the dataset (e.g., "QM9", "ZINC250k").
        kappa_ref: Reference curvature for power-law scaling.
        gamma_euler: Aggregation factor for the Euler-Maruyama solver.
        gamma_heun: Aggregation factor for the Heun solver.
        active_range: List of (start, end) time intervals where DVS is active.
            An empty list means DVS is active over the entire trajectory.
    """

    model: str
    dataset: str
    kappa_ref: float
    gamma_euler: Optional[float] = None
    gamma_heun: Optional[float] = None
    active_range: List[Tuple[float, float]] = field(default_factory=list)

    def is_active(self, time: float) -> bool:
        """Check whether DVS should be computed at the given time.

        Args:
            time: Current diffusion time in [0, T].

        Returns:
            True if ``time`` falls inside any of the configured active intervals.
        """
        if not self.active_range:
            return True
        return any(lo <= time <= hi for lo, hi in self.active_range)


# Table 7 configurations from Appendix B.1.
DATASET_CONFIGS: dict[tuple[str, str], DatasetConfig] = {
    ("GruM", "QM9"): DatasetConfig(
        model="GruM",
        dataset="QM9",
        kappa_ref=1.0,
        gamma_euler=0.22,
        gamma_heun=0.23,
        active_range=[(0.0, 1.0)],
    ),
    ("GruM", "ZINC250k"): DatasetConfig(
        model="GruM",
        dataset="ZINC250k",
        kappa_ref=5.0,
        gamma_euler=0.02,
        gamma_heun=0.04,
        active_range=[(0.0, 1.0)],
    ),
    ("GruM", "Planar"): DatasetConfig(
        model="GruM",
        dataset="Planar",
        kappa_ref=10.0,
        gamma_euler=0.31,
        gamma_heun=0.30,
        active_range=[(0.5, 1.0)],
    ),
    ("GruM", "SBM"): DatasetConfig(
        model="GruM",
        dataset="SBM",
        kappa_ref=10.0,
        gamma_euler=0.26,
        gamma_heun=0.26,
        active_range=[(0.4, 1.0)],
    ),
    ("GDSS", "QM9"): DatasetConfig(
        model="GDSS",
        dataset="QM9",
        kappa_ref=1.0,
        gamma_euler=0.68,
        gamma_heun=None,
        active_range=[(0.0, 0.2), (0.95, 1.0)],
    ),
    ("GDSS", "Ego-small"): DatasetConfig(
        model="GDSS",
        dataset="Ego-small",
        kappa_ref=0.2,
        gamma_euler=None,
        gamma_heun=None,
        active_range=[(0.0, 1.0)],
    ),
    ("GDSS", "Grid"): DatasetConfig(
        model="GDSS",
        dataset="Grid",
        kappa_ref=0.1,
        gamma_euler=None,
        gamma_heun=None,
        active_range=[(0.0, 1.0)],
    ),
    ("GDSS", "Community-small"): DatasetConfig(
        model="GDSS",
        dataset="Community-small",
        kappa_ref=0.1,
        gamma_euler=None,
        gamma_heun=None,
        active_range=[(0.0, 1.0)],
    ),
}


def get_dataset_config(model: str, dataset: str) -> DatasetConfig:
    """Retrieve a pre-defined dataset configuration from Table 7.

    Args:
        model: Backbone model name, e.g. ``"GruM"`` or ``"GDSS"``.
        dataset: Dataset name, e.g. ``"QM9"`` or ``"Planar"``.

    Returns:
        The matching ``DatasetConfig`` instance.

    Raises:
        KeyError: If the ``(model, dataset)`` pair is not present in ``DATASET_CONFIGS``.
    """
    key = (model, dataset)
    if key not in DATASET_CONFIGS:
        available = list(DATASET_CONFIGS.keys())
        raise KeyError(f"No config for {model}/{dataset}. Available: {available}")
    return DATASET_CONFIGS[key]
