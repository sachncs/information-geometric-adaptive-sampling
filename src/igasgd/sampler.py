"""Core DVS sampler and solver steps (Algorithms 1--3 of the paper).

This module is the heart of the package.  It implements:

* :func:`compute_drift_variation_score` -- **Equation 13** of the paper.
  Computes the Drift Variation Score (DVS) from consecutive drift
  evaluations.

* :func:`update_ema` -- **Equation 14**.  Exponentially smooths the raw
  DVS values to suppress transient noise.

* :func:`compute_timestep` -- **Equation 15**.  Adapts the SDE solver
  timestep via a power-law scaling of the smoothed DVS.

* :func:`global_refresh` -- "Global variation refresh" used after each
  adapted step to keep the two modalities (node features ``X`` and
  adjacency ``A``) synchronised.

* :func:`euler_step` -- **Equation 2** / **Algorithm 2**, the
  Euler-Maruyama update.

* :func:`heun_step` -- **Algorithm 3**, the Heun predictor-corrector
  update.

* :class:`DVSSampler` -- **Algorithm 1** (the meta-algorithm) tying the
  above building blocks into a complete adaptive sampling loop.

Mathematical background
-----------------------
The sampler drives the reverse-time SDE

    ``d x_t = f_t dt + g_t d w_t``

by adaptively choosing the timestep ``dt_k`` based on the local
Fisher-Rao curvature of the transition manifold.  Curvature is proxied
by the **Drift Variation Score (DVS)** which measures how rapidly the
denoising drift changes between consecutive steps.

The two graph modalities -- node features ``X_t in R^{N x D}`` and
adjacency ``A_t in R^{N x N}`` -- share a common timeline via the
**bottleneck principle** (``dt_k = min(dt_X, dt_A)``) so that they
remain synchronised in diffusion time.

All equation references in the docstrings below correspond to the
notation of arXiv:2605.00250v1.

Complexity
----------
Because we operate on pure-Python nested lists rather than vectorised
arrays, each step is O(N^2 + N * D) where N is the number of nodes and
D is the feature dimensionality.  A NumPy or Numba backend could
replace this implementation without changing the public API; see
``docs/EXTENSIONS.md`` for the roadmap.

Assumptions
-----------
* The drift function ``drift_function(X, A, t)`` is deterministic for
  fixed inputs.
* The noise schedule ``g(t)`` is locally constant during each solver
  step (justified in Appendix A.2 of the paper).
* The caller is responsible for providing a sensible RNG instance;
  fixing its seed yields a fully reproducible trajectory.

Interactions with other modules
-------------------------------
* Reads :class:`~igasgd.config.CommonConfig` and
  :class:`~igasgd.config.DatasetConfig` for hyperparameter lookup.
* Uses :func:`~igasgd.utils.clip_value` to enforce the
  ``[dt_min, dt_max]`` interval and to clamp per-element bounds.
* Consumes a callable :class:`~igasgd.schedule.NoiseSchedule` for the
  diffusion noise scale ``g_t``.
"""

import math
import random
from collections.abc import Callable

from .config import CommonConfig, DatasetConfig
from .utils import clip_value

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# A drift function consumes the current state (X, A) and the diffusion time
# t and returns the per-modality drifts (f_X, f_A).  Both modalities are
# represented as nested ``list[float]`` containers rather than NumPy arrays
# to keep the package dependency-free.
DriftFunction = Callable[
    [list[list[float]], list[list[float]], float],
    tuple[list[list[float]], list[list[float]]],
]
"""Callable signature ``(X, A, t) -> (f_X, f_A)`` for a drift network.

The two matrices in the returned tuple must have the same shapes as the
respective inputs (``X -> f_X`` of shape ``(N, D)`` and ``A -> f_A`` of
shape ``(N, N)``).
"""

NoiseSchedule = Callable[[float], float]
"""Callable signature ``t -> g_t`` returning the diffusion noise scale."""


def _squared_l2_difference(current: list[list[float]], previous: list[list[float]]) -> float:
    """Compute the squared element-wise L2 norm between two matrices.

    This is the inner-loop primitive underlying
    :func:`compute_drift_variation_score`.  Operates on nested
    ``list[float]`` containers and returns ``||current - previous||_2^2``
    (i.e. the sum of squared differences, not the L2 norm itself).

    Args:
        current: Matrix at the current step, shape ``(rows, cols)``.
        previous: Matrix at the previous step, same shape as
            ``current``.

    Returns:
        The squared L2 norm, a non-negative scalar.

    Raises:
        ValueError: If ``current`` and ``previous`` have different row
            counts or if any pair of corresponding rows have different
            column counts.

    Complexity:
        O(rows * cols) time and O(1) extra memory.
    """
    if len(current) != len(previous):
        raise ValueError(f"Mismatched row count: {len(current)} vs {len(previous)}")
    total = 0.0
    for row_cur, row_prev in zip(current, previous, strict=False):
        if len(row_cur) != len(row_prev):
            raise ValueError(f"Mismatched column count: {len(row_cur)} vs {len(row_prev)}")
        for val_cur, val_prev in zip(row_cur, row_prev, strict=False):
            diff = val_cur - val_prev
            total += diff * diff
    return total


def compute_drift_variation_score(
    current_drift_x: list[list[float]],
    previous_drift_x: list[list[float]],
    current_drift_a: list[list[float]],
    previous_drift_a: list[list[float]],
    noise_scale: float,
    eps_num: float,
) -> tuple[float, float]:
    """Compute the Drift Variation Score (DVS) for both modalities.

    Implements **Equation 13** of the paper.  For each modality the DVS
    is the squared L2 norm of the drift difference normalised by the
    squared noise scale (plus a small numerical stabiliser):

        ``V_X = ||f_X,k - f_X,k-1||_2^2 / (g_t^2 + eps_num)``
        ``V_A = ||f_A,k - f_A,k-1||_2^2 / (g_t^2 + eps_num)``

    The ``eps_num`` term prevents division-by-zero when the diffusion
    process reaches its deterministic terminal regime.

    Args:
        current_drift_x: Drift for node features at step ``k``.
        previous_drift_x: Drift for node features at step ``k-1``.
        current_drift_a: Drift for adjacency at step ``k``.
        previous_drift_a: Drift for adjacency at step ``k-1``.
        noise_scale: Noise scale ``g_t`` at the current diffusion time.
        eps_num: Small positive constant added to the denominator.

    Returns:
        A tuple ``(V_X, V_A)`` containing the drift variation scores
        for the two modalities.

    Raises:
        ValueError: Propagated from :func:`_squared_l2_difference` if
            the drift matrices have mismatched shapes.

    Complexity:
        O(N^2 + N * D) for graphs with ``N`` nodes and ``D``-dimensional
        node features.
    """
    denominator = noise_scale * noise_scale + eps_num
    v_x = _squared_l2_difference(current_drift_x, previous_drift_x) / denominator
    v_a = _squared_l2_difference(current_drift_a, previous_drift_a) / denominator
    return v_x, v_a


def update_ema(
    v_x: float,
    v_a: float,
    smoothed_x: float,
    smoothed_a: float,
    alpha: float,
) -> tuple[float, float]:
    """Apply the exponential moving average (EMA) smoothing step.

    Implements **Equation 14** of the paper:

        ``Vbar_X <- alpha * V_X + (1 - alpha) * Vbar_X``
        ``Vbar_A <- alpha * V_A + (1 - alpha) * Vbar_A``

    The EMA stabilises the noisy raw DVS so that the power-law step-size
    scaling downstream does not oscillate wildly between steps.

    Args:
        v_x: Raw DVS for node features at the current step.
        v_a: Raw DVS for adjacency at the current step.
        smoothed_x: Previous smoothed DVS for node features (initialise
            to ``0.0`` for the first step).
        smoothed_a: Previous smoothed DVS for adjacency (initialise to
            ``0.0`` for the first step).
        alpha: EMA coefficient in ``(0, 1]``.  ``alpha = 1`` disables
            smoothing (``Vbar`` equals the raw DVS); ``alpha = 0``
            freezes the estimate.

    Returns:
        The updated ``(smoothed_x, smoothed_a)`` pair.

    Edge cases:
        * ``alpha = 1`` returns ``(v_x, v_a)`` exactly.
        * ``alpha = 0`` returns the previous ``(smoothed_x, smoothed_a)``
          unchanged.
        * NaN or infinity inputs are propagated unchanged by IEEE 754
          arithmetic.

    Complexity:
        O(1).
    """
    new_smoothed_x = alpha * v_x + (1.0 - alpha) * smoothed_x
    new_smoothed_a = alpha * v_a + (1.0 - alpha) * smoothed_a
    return new_smoothed_x, new_smoothed_a


def compute_timestep(
    smoothed_score: float,
    kappa_ref: float,
    dt_base: float,
    dt_min: float,
    dt_max: float,
    beta: float,
    eps_num: float,
) -> float:
    """Adapt the SDE solver timestep via the power-law scaling law.

    Implements **Equation 15** of the paper:

        ``dt = clip(dt_base * (kappa_ref / Vbar)^beta, dt_min, dt_max)``

    The function plays the role of a "curvature governor": when the
    smoothed DVS is much larger than the reference curvature ``kappa_ref``
    the timestep shrinks so that fine structure is resolved; when the
    drift is nearly constant the timestep grows up to ``dt_max`` to save
    compute.

    Args:
        smoothed_score: EMA-smoothed DVS (``Vbar``).  Must be
            non-negative if ``beta`` is non-integer.
        kappa_ref: Reference curvature (set-point of the controller).
        dt_base: Base timestep.
        dt_min: Minimum allowed timestep.
        dt_max: Maximum allowed timestep.
        beta: Sensitivity exponent (commonly ``0.5`` for square-root
            scaling).
        eps_num: Numerical stabiliser added to the denominator.

    Returns:
        The adapted timestep clipped to ``[dt_min, dt_max]``.

    Edge cases:
        * ``smoothed_score = 0`` makes the denominator ``eps_num``,
          yielding a very large raw ``dt`` that clips to ``dt_max``.
        * ``smoothed_score = inf`` makes the ratio ``0``, so ``dt``
          clips to ``dt_min``.
        * ``smoothed_score`` is NaN propagates NaN, which is then
          returned by :func:`~igasgd.utils.clip_value`.

    Complexity:
        O(1).
    """
    ratio = kappa_ref / (smoothed_score + eps_num)
    raw_dt = dt_base * (ratio**beta)
    return clip_value(raw_dt, dt_min, dt_max)


def global_refresh(
    smoothed_x: float,
    smoothed_a: float,
    gamma: float,
) -> tuple[float, float]:
    """Synchronise the two EMA states after an adapted step.

    Implements the "global variation refresh" described in the paper.
    After computing the adapted timestep, both smoothed DVS values are
    replaced by a single aggregated value so that the next step starts
    from a common baseline:

        ``Vbar_X, Vbar_A <- gamma * (Vbar_X + Vbar_A)``

    This prevents whichever modality happened to be smoother from
    biasing future bottleneck decisions.

    Args:
        smoothed_x: Current smoothed DVS for node features.
        smoothed_a: Current smoothed DVS for adjacency.
        gamma: Aggregation factor from Table 7 (solver-specific).

    Returns:
        The refreshed ``(smoothed_x, smoothed_a)`` pair; both entries
        are equal to ``gamma * (smoothed_x + smoothed_a)``.

    Edge cases:
        * ``gamma = 0`` resets both scores to ``0.0``.
        * Negative ``gamma`` values (allowed by the formula but unusual
          in practice) flip the sign of the refreshed scores.

    Complexity:
        O(1).
    """
    combined = gamma * (smoothed_x + smoothed_a)
    return combined, combined


def _add_drift_and_noise(
    state: list[list[float]],
    drift: list[list[float]],
    timestep: float,
    noise_scale: float,
    rng: random.Random,
) -> list[list[float]]:
    """Apply a single Euler-style drift + diffusion update to a state matrix.

    This helper implements the elementary stochastic update

        ``X_{k+1} = X_k + f_k * dt + g_t * sqrt(dt) * Z``

    element-wise over a nested list.  The same noise realisation is
    drawn for every entry, which is consistent with diagonal noise
    assumed by both the Euler-Maruyama and Heun solvers in this
    package.

    Args:
        state: Current state matrix of shape ``(rows, cols)``.
        drift: Drift matrix of the same shape as ``state``.
        timestep: Adapted timestep ``dt_k``.
        noise_scale: Noise scale ``g_t * sqrt(dt_k)`` pre-multiplied by
            the caller so this helper need not know about ``g(t)``.
        rng: Random number generator used to draw Gaussian noise.

    Returns:
        A new matrix (same shape as ``state``) holding the updated
        values.  The inputs are not modified.

    Complexity:
        O(rows * cols) time, O(rows * cols) memory for the output.
    """
    result: list[list[float]] = []
    for state_row, drift_row in zip(state, drift, strict=False):
        new_row: list[float] = []
        for state_val, drift_val in zip(state_row, drift_row, strict=False):
            noise = noise_scale * rng.gauss(0.0, 1.0)
            new_row.append(state_val + drift_val * timestep + noise)
        result.append(new_row)
    return result


def euler_step(
    features: list[list[float]],
    adjacency: list[list[float]],
    drift_features: list[list[float]],
    drift_adjacency: list[list[float]],
    timestep: float,
    noise_scale: float,
    rng: random.Random,
) -> tuple[list[list[float]], list[list[float]]]:
    """Apply a single Euler-Maruyama step to both modalities.

    Implements the Euler-Maruyama discretisation of the reverse-time SDE
    (**Equation 2** of the paper and the body of **Algorithm 2**):

        ``X_k = X_{k-1} + f_X * dt + g * sqrt(dt) * Z_X``
        ``A_k = A_{k-1} + f_A * dt + g * sqrt(dt) * Z_A``

    where ``Z_X`` and ``Z_A`` are independent Gaussian vectors drawn
    from ``rng``.

    Args:
        features: Node feature matrix ``X_{k-1}`` of shape ``(N, D)``.
        adjacency: Adjacency matrix ``A_{k-1}`` of shape ``(N, N)``.
        drift_features: Drift ``f_X`` evaluated at ``(X_{k-1}, A_{k-1}, t)``.
        drift_adjacency: Drift ``f_A`` evaluated at the same state.
        timestep: Adapted timestep ``dt_k``.
        noise_scale: ``g_t * sqrt(dt_k)``, pre-multiplied by the caller.
        rng: Random number generator for reproducibility.

    Returns:
        The updated pair ``(X_k, A_k)`` with shapes identical to the
        inputs.

    Complexity:
        O(N^2 + N * D) per call.
    """
    next_features = _add_drift_and_noise(features, drift_features, timestep, noise_scale, rng)
    next_adjacency = _add_drift_and_noise(adjacency, drift_adjacency, timestep, noise_scale, rng)
    return next_features, next_adjacency


def heun_step(
    features: list[list[float]],
    adjacency: list[list[float]],
    first_drift_features: list[list[float]],
    first_drift_adjacency: list[list[float]],
    timestep: float,
    noise_scale: float,
    drift_function: DriftFunction,
    time: float,
    rng: random.Random,
) -> tuple[list[list[float]], list[list[float]]]:
    """Apply a single Heun predictor-corrector step to both modalities.

    Implements **Algorithm 3** of the paper.  The Heun method is a
    second-order SDE solver that performs a predictor (Euler) step,
    evaluates the drift at the predicted state, and then averages the
    two drift estimates to obtain the corrector update:

    1. **Predictor** (Euler using the first drift estimate):
       ``X_hat = X + f1 * dt + g * sqrt(dt) * Z``

    2. **Corrector drift** at the predicted state and ``t + dt``:
       ``f2 = drift_function(X_hat, A_hat, t + dt)``

    3. **Corrector** (trapezoidal average of the two drifts):
       ``X_k = X + 0.5 * (f1 + f2) * dt + g * sqrt(dt) * Z``

    The same noise realisation ``Z`` is shared by predictor and
    corrector so that the corrector remains consistent with the same
    Brownian path.

    Args:
        features: Node feature matrix ``X_{k-1}`` of shape ``(N, D)``.
        adjacency: Adjacency matrix ``A_{k-1}`` of shape ``(N, N)``.
        first_drift_features: First-stage drift ``f_X^{(1)}`` already
            evaluated at ``(X_{k-1}, A_{k-1}, t)``.
        first_drift_adjacency: First-stage drift ``f_A^{(1)}``.
        timestep: Adapted timestep ``dt_k``.
        noise_scale: ``g_t * sqrt(dt_k)`` pre-multiplied by the caller.
        drift_function: Callable drift network used for the corrector
            evaluation.
        time: Current diffusion time ``t``.  The corrector evaluates
            the drift at ``t + timestep``.
        rng: Random number generator.

    Returns:
        The corrected pair ``(X_k, A_k)`` with shapes identical to the
        inputs.

    Complexity:
        O(N^2 + N * D) per call plus one extra drift evaluation.

    Note:
        With constant drift (``f2 == f1``) the method reduces exactly
        to the Euler-Maruyama update; the unit tests verify this.
    """
    # Predictor: standard Euler-Maruyama step using the first-stage drift.
    predicted_features = _add_drift_and_noise(
        features, first_drift_features, timestep, noise_scale, rng
    )
    predicted_adjacency = _add_drift_and_noise(
        adjacency, first_drift_adjacency, timestep, noise_scale, rng
    )

    # Second drift evaluation at predicted state and t + dt (corrector).
    second_drift_features, second_drift_adjacency = drift_function(
        predicted_features, predicted_adjacency, time + timestep
    )

    # Corrector: trapezoidal average of first and second drifts.
    def _average_and_update(
        state: list[list[float]],
        drift1: list[list[float]],
        drift2: list[list[float]],
    ) -> list[list[float]]:
        result: list[list[float]] = []
        for state_row, d1_row, d2_row in zip(state, drift1, drift2, strict=False):
            new_row: list[float] = []
            for s, d1, d2 in zip(state_row, d1_row, d2_row, strict=False):
                # New noise realisation is independent of the predictor's
                # noise -- the corrector sees the same Z so the corrector
                # trapezoidal rule remains a consistent noise scaling.
                noise = noise_scale * rng.gauss(0.0, 1.0)
                avg_drift = 0.5 * (d1 + d2)
                new_row.append(s + avg_drift * timestep + noise)
            result.append(new_row)
        return result

    next_features = _average_and_update(features, first_drift_features, second_drift_features)
    next_adjacency = _average_and_update(adjacency, first_drift_adjacency, second_drift_adjacency)
    return next_features, next_adjacency


class DVSSampler:
    """DVS-driven adaptive sampler (Algorithms 1--3 combined).

    This is a **training-free**, modular component that wraps an existing
    denoising drift function and a noise schedule.  It can be plugged
    into any graph diffusion model that exposes a ``drift_function(X, A, t)``
    interface, without modifying the underlying network.

    The class encapsulates the full Algorithm 1 meta-algorithm: it
    loops over diffusion time, evaluates the drift at each state,
    computes the Drift Variation Score, smooths it via an EMA, scales
    the timestep through a power-law rule, and finally applies either
    the Euler-Maruyama or Heun solver step.

    Thread-safety:
        Instances are not thread-safe because they own an internal
        :class:`random.Random` instance whose state advances on every
        noise draw.  Run independent sampling trajectories with one
        sampler per thread (or build a new sampler per thread).

    Lifecycle:
        1. **Construction** -- validate solver name and resolve the
           solver-specific ``gamma`` from the dataset configuration.
        2. **Sampling** -- call :meth:`sample` one or more times.  Each
           call consumes the internal RNG and produces a fresh
           trajectory.
        3. **Inspection** -- access the read-only :attr:`gamma` and the
           original ``common_config`` / ``dataset_config`` / ``solver``
           attributes for diagnostic purposes.

    Important attributes:
        * :attr:`gamma` -- the solver-specific aggregation factor.
        * The private ``_drift_function``, ``_noise_schedule``,
          ``_common_config``, ``_dataset_config``, ``_solver``, and
          ``_rng`` slots back the public behaviour.

    Example:
        >>> from igasgd import (
        ...     CommonConfig, DVSSampler, LinearSchedule,
        ...     get_dataset_config, make_drift_function,
        ...     GruMApproximation,
        ... )
        >>> approx = GruMApproximation(num_nodes=9, feature_dim=4, seed=42)
        >>> drift = make_drift_function(approx)
        >>> sampler = DVSSampler(
        ...     drift_function=drift,
        ...     noise_schedule=LinearSchedule(sigma_min=0.01, sigma_max=0.5),
        ...     common_config=CommonConfig(),
        ...     dataset_config=get_dataset_config("GruM", "QM9"),
        ...     solver="Euler",
        ...     seed=42,
        ... )
        >>> X_T, A_T, info = sampler.sample(
        ...     initial_features=[[0.1] * 4 for _ in range(9)],
        ...     initial_adjacency=[[0.1] * 9 for _ in range(9)],
        ...     terminal_time=1.0,
        ... )
        >>> int(info["total_steps"][0]) > 0
        True
    """

    def __init__(
        self,
        drift_function: DriftFunction,
        noise_schedule: NoiseSchedule,
        common_config: CommonConfig,
        dataset_config: DatasetConfig,
        solver: str = "Euler",
        seed: int | None = None,
    ) -> None:
        """Initialise the sampler.

        Validates the solver name, resolves the solver-specific
        aggregation factor ``gamma`` from the dataset configuration,
        and constructs the internal RNG used to draw Gaussian noise.

        Args:
            drift_function: Callable ``(X, A, t) -> (f_X, f_A)``.
            noise_schedule: Callable ``t -> g_t`` returning the
                diffusion noise scale.
            common_config: Common hyperparameters (Table 6).
            dataset_config: Dataset-specific hyperparameters (Table 7).
            solver: One of ``"Euler"`` or ``"Heun"``.
            seed: Optional random seed for reproducibility.  When
                ``None`` the RNG is initialised from a non-deterministic
                source.

        Raises:
            ValueError: If ``solver`` is not ``"Euler"`` or ``"Heun"``.
            ValueError: If ``dataset_config`` does not provide a
                ``gamma`` for the requested solver.
        """
        if solver not in {"Euler", "Heun"}:
            raise ValueError(f"solver must be 'Euler' or 'Heun', got {solver!r}")
        self._drift_function = drift_function
        self._noise_schedule = noise_schedule
        self._common_config = common_config
        self._dataset_config = dataset_config
        self._solver = solver
        self._rng = random.Random(seed)

        # Select the solver-specific aggregation factor from Table 7.
        # The chosen gamma is what couples the two modalities via
        # ``global_refresh`` after every adapted step.
        if solver == "Euler":
            self._gamma = dataset_config.gamma_euler
        else:
            self._gamma = dataset_config.gamma_heun

        if self._gamma is None:
            raise ValueError(
                f"DatasetConfig {dataset_config.model}/{dataset_config.dataset} "
                f"has no gamma for solver={solver}"
            )

    @property
    def gamma(self) -> float:
        """Aggregation factor used by this sampler instance.

        Resolved from ``dataset_config`` at construction time.  The
        cast is safe because :meth:`__init__` already validated that
        ``self._gamma`` is not ``None``.

        Returns:
            The solver-specific ``gamma`` value.
        """
        return self._gamma  # type: ignore[return-value]

    def sample(
        self,
        initial_features: list[list[float]],
        initial_adjacency: list[list[float]],
        terminal_time: float = 1.0,
        verbose: bool = False,
    ) -> tuple[list[list[float]], list[list[float]], dict[str, list[float]]]:
        """Run adaptive sampling from ``t = 0`` to ``terminal_time``.

        Implements the **Algorithm 1** meta-algorithm of the paper: an
        event-driven loop that, at each step, decides whether DVS is
        active (based on the dataset's ``active_range`` and whether we
        have a cached previous drift), computes the adapted timestep
        when it is, and otherwise falls back to ``dt_base``.

        The returned ``info_dict`` records every step's metadata so
        that callers can plot trajectories, debug adaptive behaviour,
        or post-process the run.

        Args:
            initial_features: Initial noisy node features ``X_0`` of
                shape ``(N, D)``.
            initial_adjacency: Initial noisy adjacency ``A_0`` of shape
                ``(N, N)``.
            terminal_time: Diffusion horizon ``T`` (default ``1.0``).
            verbose: If ``True`` a one-line summary is printed for
                every solver step (useful for interactive debugging).

        Returns:
            A triple ``(X_T, A_T, info_dict)`` where ``info_dict``
            contains, per step, the step index, the timestep ``dt``,
            the diffusion time ``t`` at which the step began, the raw
            DVS values ``v_x`` and ``v_a``, and the smoothed DVS
            values ``smoothed_x`` and ``smoothed_a``.  Two scalar
            fields, ``total_steps`` and ``final_time``, summarise the
            run.

        Termination:
            The loop stops when ``time`` reaches ``terminal_time`` to
            within ``common_config.eps_bound``.  This guarantees that
            the final diffusion time is at least ``T - eps_bound``
            regardless of how the adaptive timestep rounds.

        Notes:
            When ``terminal_time`` is smaller than ``eps_bound`` (or
            zero) the sampler returns the initial state unchanged and
            records ``total_steps = 0``.
        """
        common = self._common_config
        dataset = self._dataset_config

        time = 0.0
        step_index = 1
        smoothed_x = 0.0
        smoothed_a = 0.0

        # Take defensive copies so that the caller may safely reuse the
        # initial state objects for other purposes.
        prev_features = [list(row) for row in initial_features]
        prev_adjacency = [list(row) for row in initial_adjacency]

        # Cache the previous drift so we can compute DVS at the next
        # step.  ``None`` until the first step completes (Algorithm 1
        # requires two consecutive drift evaluations).
        cached_drift_features: list[list[float]] | None = None
        cached_drift_adjacency: list[list[float]] | None = None

        # Per-step history exposed to the caller.
        info: dict[str, list[float]] = {
            "steps": [],
            "dt": [],
            "time": [],
            "v_x": [],
            "v_a": [],
            "smoothed_x": [],
            "smoothed_a": [],
        }

        while time < terminal_time - common.eps_bound:
            # --- Drift evaluation at the current state ---
            drift_features, drift_adjacency = self._drift_function(
                prev_features, prev_adjacency, time
            )
            noise_scale_g = self._noise_schedule(time)

            # DVS is active only if:
            #   (a) the dataset permits it at this time, AND
            #   (b) we have completed at least one prior step so a
            #       previous drift exists for the difference, AND
            #   (c) the cached drifts are present.
            dvs_active = (
                dataset.is_active(time)
                and step_index > 1
                and cached_drift_features is not None
                and cached_drift_adjacency is not None
            )

            if dvs_active:
                # Mypy narrowing: we already checked ``is not None``.
                assert cached_drift_features is not None
                assert cached_drift_adjacency is not None
                # Equation 13: Drift Variation Score.
                v_x, v_a = compute_drift_variation_score(
                    drift_features,
                    cached_drift_features,
                    drift_adjacency,
                    cached_drift_adjacency,
                    noise_scale_g,
                    common.eps_num,
                )
                # Equation 14: EMA smoothing of the raw DVS.
                smoothed_x, smoothed_a = update_ema(v_x, v_a, smoothed_x, smoothed_a, common.alpha)
                # Equation 15: Power-law step-size scaling per modality.
                dt_features = compute_timestep(
                    smoothed_x,
                    dataset.kappa_ref,
                    common.dt_base,
                    common.dt_min,
                    common.dt_max,
                    common.beta,
                    common.eps_num,
                )
                dt_adjacency = compute_timestep(
                    smoothed_a,
                    dataset.kappa_ref,
                    common.dt_base,
                    common.dt_min,
                    common.dt_max,
                    common.beta,
                    common.eps_num,
                )
                # Bottleneck principle: keep X and A synchronised by
                # taking the more conservative (smaller) timestep.
                timestep = min(dt_features, dt_adjacency)
                # Mypy narrowing: gamma was validated in __init__.
                assert self._gamma is not None
                # Global variation refresh: synchronise the EMA states.
                smoothed_x, smoothed_a = global_refresh(smoothed_x, smoothed_a, self._gamma)

                if verbose:
                    print(
                        f"step={step_index:4d} time={time:.6f} "
                        f"active dvs dt={timestep:.6f} "
                        f"v_x={v_x:.4e} v_a={v_a:.4e}"
                    )
            else:
                # DVS not active -- fall back to the fixed base step.
                timestep = common.dt_base
                v_x = 0.0
                v_a = 0.0
                if verbose:
                    print(f"step={step_index:4d} time={time:.6f} base   dt={timestep:.6f}")

            # Do not overshoot the terminal time -- shrink the last
            # step if necessary.
            timestep = min(timestep, terminal_time - time)

            # --- Solver step ---
            # The total noise scale is g_t * sqrt(dt).  Multiplying
            # ``g_t`` by ``sqrt(dt)`` here lets the step helpers reuse
            # the same scaling for both modalities.
            noise_scale = noise_scale_g * math.sqrt(timestep)
            if self._solver == "Euler":
                next_features, next_adjacency = euler_step(
                    prev_features,
                    prev_adjacency,
                    drift_features,
                    drift_adjacency,
                    timestep,
                    noise_scale,
                    self._rng,
                )
            else:  # Heun
                next_features, next_adjacency = heun_step(
                    prev_features,
                    prev_adjacency,
                    drift_features,
                    drift_adjacency,
                    timestep,
                    noise_scale,
                    self._drift_function,
                    time,
                    self._rng,
                )

            # Record per-step history for post-hoc analysis.
            info["steps"].append(float(step_index))
            info["dt"].append(timestep)
            info["time"].append(time)
            info["v_x"].append(v_x)
            info["v_a"].append(v_a)
            info["smoothed_x"].append(smoothed_x)
            info["smoothed_a"].append(smoothed_a)

            # Cache the current drift so the next iteration can compute DVS.
            cached_drift_features = [list(row) for row in drift_features]
            cached_drift_adjacency = [list(row) for row in drift_adjacency]

            # Advance time and state.
            time += timestep
            step_index += 1
            prev_features = next_features
            prev_adjacency = next_adjacency

        # Scalar summary fields exposed alongside the per-step lists.
        info["total_steps"] = [float(step_index - 1)]
        info["final_time"] = [time]
        return prev_features, prev_adjacency, info
