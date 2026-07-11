"""Hyperparameter configurations extracted from Tables 6 and 7 of the paper.

This module centralises all constants used by the DVS-driven adaptive sampler
into two frozen dataclasses plus a global lookup table.

Architecture overview
---------------------
* :class:`CommonConfig` -- Dataset-independent hyperparameters (Table 6).
  These values were kept constant across every benchmark reported in the
  paper and are relatively robust to mild setting changes.  They govern the
  EMA smoothing of the drift variation score, the power-law step-size
  response, the admissible timestep interval, and two numerical guards
  (``eps_num``, ``eps_bound``).

* :class:`DatasetConfig` -- Dataset- and backbone-specific hyperparameters
  (Table 7).  Each entry binds together a ``kappa_ref`` reference curvature,
  the solver-specific aggregation factors ``gamma_euler`` / ``gamma_heun``,
  and the active time intervals in which DVS is computed.

* :data:`DATASET_CONFIGS` -- Module-level mapping from ``(model, dataset)``
  pairs to pre-built :class:`DatasetConfig` instances.  The lookup helper
  :func:`get_dataset_config` performs validation and surfaces an informative
  error when a pair is missing.

Why frozen dataclasses?
-----------------------
Both classes are declared ``frozen=True`` so that configuration instances can
be safely shared across threads and processes without risk of accidental
mutation.  Default values match the paper exactly so that downstream code
can instantiate them with no arguments for paper-faithful reproduction.

Interactions with other modules
-------------------------------
This module is consumed by :mod:`igasgd.sampler` to construct
:class:`~igasgd.sampler.DVSSampler` instances.  It does not depend on any
other module of the package.

References:
----------
* Paper Section 4 (Table 6) and Appendix B.1 (Table 7).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommonConfig:
    """Common hyperparameters for the DVS-driven adaptive sampler.

    These values are kept constant across all benchmarks reported in the
    paper and are relatively robust to mild setting changes.  The defaults
    below reproduce the paper's Table 6 exactly.

    The dataclass is frozen so that a configuration instance can be safely
    shared across threads; attempts to mutate any field will raise
    ``dataclasses.FrozenInstanceError``.

    Attributes:
        alpha: EMA smoothing coefficient used in Equation 14.  A small
            ``alpha`` (e.g. ``0.2``) produces a slow, stable estimate of
            the drift variation score; values close to ``1.0`` discard
            the history entirely.
        beta: Sensitivity exponent in Equation 15 controlling how strongly
            the adapted timestep reacts to the smoothed DVS.  ``beta=0.5``
            (the paper's choice) corresponds to a square-root response.
        dt_base: Reference timestep size in diffusion time units.  Used as
            the initial timestep when DVS is inactive and as the
            multiplicative base in the power-law scaling law.
        dt_min: Lower bound on the adapted timestep.  Prevents the sampler
            from stalling in high-curvature regions where the power law
            would otherwise produce vanishingly small steps.
        dt_max: Upper bound on the adapted timestep.  Maintains numerical
            stability when the drift variation collapses.
        eps_num: Small positive constant added to denominators to prevent
            division-by-zero.  Used both in the DVS computation
            (``g_t^2 + eps_num``) and in the timestep scaling law.
        eps_bound: Boundary tolerance controlling when the sampling loop
            terminates.  The loop stops when the remaining time ``T - t``
            drops below this threshold, ensuring termination at ``T`` to
            within ``eps_bound``.

    Example:
        >>> cfg = CommonConfig()
        >>> cfg.alpha, cfg.beta
        (0.2, 0.5)
        >>> cfg.dt_base
        0.001
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
    """Dataset-specific hyperparameters and adaptive DVS intervals.

    Each ``(model, dataset)`` benchmark from the paper has its own
    reference curvature ``kappa_ref`` (the set-point around which the
    power-law timestep is calibrated), an aggregation factor per solver,
    and a list of time intervals in which DVS is computed.  Outside these
    intervals the sampler falls back to a fixed ``dt_base`` timestep.

    The dataclass is frozen so that a configuration instance can be safely
    shared across threads; attempting to mutate any field raises
    ``dataclasses.FrozenInstanceError``.

    Attributes:
        model: Name of the backbone model (e.g. ``"GruM"``, ``"GDSS"``).
        dataset: Name of the dataset (e.g. ``"QM9"``, ``"ZINC250k"``).
        kappa_ref: Reference curvature for the power-law scaling.  When
            the smoothed DVS equals ``kappa_ref`` the adapted timestep
            recovers the base ``dt_base``.
        gamma_euler: Aggregation factor used by :func:`global_refresh`
            when the Euler-Maruyama solver is selected.  ``None`` if the
            dataset does not support the Euler solver.
        gamma_heun: Aggregation factor used by :func:`global_refresh`
            when the Heun solver is selected.  ``None`` if the dataset
            does not support the Heun solver.
        active_range: List of ``(start, end)`` time intervals during
            which DVS is computed.  An empty list means that DVS is
            active over the entire trajectory.  Intervals are inclusive
            on both endpoints.

    Example:
        >>> cfg = DatasetConfig(
        ...     model="GruM",
        ...     dataset="QM9",
        ...     kappa_ref=1.0,
        ...     gamma_euler=0.22,
        ...     gamma_heun=0.23,
        ...     active_range=[(0.0, 1.0)],
        ... )
        >>> cfg.is_active(0.5)
        True
    """

    model: str
    dataset: str
    kappa_ref: float
    gamma_euler: float | None = None
    gamma_heun: float | None = None
    active_range: list[tuple[float, float]] = field(default_factory=list)

    def is_active(self, time: float) -> bool:
        """Check whether DVS should be computed at the given diffusion time.

        The check is inclusive on both interval endpoints so that a time
        exactly equal to ``start`` or ``end`` is treated as inside the
        active range.  An empty ``active_range`` is interpreted as
        "always active", matching the convention used by ``in_active_range``
        in :mod:`igasgd.utils`.

        Args:
            time: Current diffusion time ``t`` in ``[0, T]``.

        Returns:
            ``True`` if ``time`` falls inside at least one configured
            active interval (or if no intervals were configured);
            otherwise ``False``.

        Complexity:
            Linear in the number of configured intervals.
        """
        if not self.active_range:
            return True
        return any(lo <= time <= hi for lo, hi in self.active_range)


# Table 7 configurations from Appendix B.1.
#
# Each entry below mirrors a row of Table 7 in the paper.  The active_range
# intervals are inclusive on both endpoints, matching the convention used
# by ``DatasetConfig.is_active`` and ``utils.in_active_range``.
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

    The lookup is performed against the module-level
    :data:`DATASET_CONFIGS` dictionary.  If the requested pair is absent,
    a ``KeyError`` is raised with an actionable message that lists every
    available key.

    Args:
        model: Backbone model name, e.g. ``"GruM"`` or ``"GDSS"``.
        dataset: Dataset name, e.g. ``"QM9"`` or ``"Planar"``.

    Returns:
        The matching :class:`DatasetConfig` instance.

    Raises:
        KeyError: If the ``(model, dataset)`` pair is not present in
            :data:`DATASET_CONFIGS`.  The exception message includes the
            list of available pairs to aid debugging.

    Example:
        >>> cfg = get_dataset_config("GruM", "QM9")
        >>> cfg.kappa_ref
        1.0
    """
    key = (model, dataset)
    if key not in DATASET_CONFIGS:
        available = list(DATASET_CONFIGS.keys())
        raise KeyError(f"No config for {model}/{dataset}. Available: {available}")
    return DATASET_CONFIGS[key]
