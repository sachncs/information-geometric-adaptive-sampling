"""Core DVS sampler and solver steps.

This module implements the three algorithms from the paper:

- Algorithm 1: DVS-driven adaptive sampler (meta-algorithm).
- Algorithm 2: DVS-Euler--Maruyama adaptive sampler.
- Algorithm 3: DVS-Heun adaptive sampler.

All equations referenced in docstrings correspond to arXiv:2605.00250v1.
"""

import math
import random
from typing import Callable, Dict, List, Optional, Tuple

from .config import CommonConfig, DatasetConfig
from .utils import clip_value

# Type aliases for readability
DriftFunction = Callable[
    [List[List[float]], List[List[float]], float],
    Tuple[List[List[float]], List[List[float]]],
]
NoiseSchedule = Callable[[float], float]


def _squared_l2_difference(
    current: List[List[float]], previous: List[List[float]]
) -> float:
    """Compute ||current - previous||_2^2 element-wise over nested lists.

    Raises:
        ValueError: If the two matrices have mismatched shapes.
    """
    if len(current) != len(previous):
        raise ValueError(f"Mismatched row count: {len(current)} vs {len(previous)}")
    total = 0.0
    for row_cur, row_prev in zip(current, previous):
        if len(row_cur) != len(row_prev):
            raise ValueError(
                f"Mismatched column count: {len(row_cur)} vs {len(row_prev)}"
            )
        for val_cur, val_prev in zip(row_cur, row_prev):
            diff = val_cur - val_prev
            total += diff * diff
    return total


def compute_drift_variation_score(
    current_drift_x: List[List[float]],
    previous_drift_x: List[List[float]],
    current_drift_a: List[List[float]],
    previous_drift_a: List[List[float]],
    noise_scale: float,
    eps_num: float,
) -> Tuple[float, float]:
    """Equation 13: Drift Variation Score (DVS).

    For each modality (node features X and adjacency A) the DVS is defined as
    the squared L2 norm of the drift difference divided by the squared noise
    scale (plus a numerical stabiliser):

        V_X = ||f_X,k - f_X,k-1||_2^2 / (g_t^2 + eps_num)
        V_A = ||f_A,k - f_A,k-1||_2^2 / (g_t^2 + eps_num)

    Args:
        current_drift_x: Drift for node features at step k.
        previous_drift_x: Drift for node features at step k-1.
        current_drift_a: Drift for adjacency at step k.
        previous_drift_a: Drift for adjacency at step k-1.
        noise_scale: Noise scale g_t at the current time.
        eps_num: Small constant preventing division-by-zero.

    Returns:
        A tuple ``(V_X, V_A)`` containing the drift variation scores.
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
) -> Tuple[float, float]:
    """Equation 14: Exponential Moving Average (EMA) smoothing.

        Vbar_X <- alpha * V_X + (1 - alpha) * Vbar_X
        Vbar_A <- alpha * V_A + (1 - alpha) * Vbar_A

    Args:
        v_x: Raw DVS for node features at the current step.
        v_a: Raw DVS for adjacency at the current step.
        smoothed_x: Previous smoothed DVS for node features.
        smoothed_a: Previous smoothed DVS for adjacency.
        alpha: EMA coefficient in (0, 1).

    Returns:
        Updated ``(smoothed_x, smoothed_a)``.
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
    """Equation 15: Power-law step-size scaling.

        dt = clip(dt_base * (kappa_ref / Vbar)^beta, dt_min, dt_max)

    Args:
        smoothed_score: EMA-smoothed DVS (Vbar).
        kappa_ref: Reference curvature.
        dt_base: Base timestep.
        dt_min: Minimum allowed timestep.
        dt_max: Maximum allowed timestep.
        beta: Sensitivity exponent.
        eps_num: Numerical stabiliser for the denominator.

    Returns:
        The adapted timestep clipped to ``[dt_min, dt_max]``.
    """
    ratio = kappa_ref / (smoothed_score + eps_num)
    raw_dt = dt_base * (ratio**beta)
    return clip_value(raw_dt, dt_min, dt_max)


def global_refresh(
    smoothed_x: float,
    smoothed_a: float,
    gamma: float,
) -> Tuple[float, float]:
    """Global variation refresh after computing the timestep.

    Both smoothed scores are replaced by the same aggregated value so that
    subsequent steps start from a synchronised state:

        Vbar_X, Vbar_A <- gamma * (Vbar_X + Vbar_A)

    Args:
        smoothed_x: Current smoothed DVS for node features.
        smoothed_a: Current smoothed DVS for adjacency.
        gamma: Aggregation factor from Table 7.

    Returns:
        The refreshed ``(smoothed_x, smoothed_a)`` pair (both equal).
    """
    combined = gamma * (smoothed_x + smoothed_a)
    return combined, combined


def _add_drift_and_noise(
    state: List[List[float]],
    drift: List[List[float]],
    timestep: float,
    noise_scale: float,
    rng: random.Random,
) -> List[List[float]]:
    """Apply drift and scaled Gaussian noise to a state matrix.

    This is the elementary update shared by both Euler and Heun steps.
    """
    result: List[List[float]] = []
    for state_row, drift_row in zip(state, drift):
        new_row: List[float] = []
        for state_val, drift_val in zip(state_row, drift_row):
            noise = noise_scale * rng.gauss(0.0, 1.0)
            new_row.append(state_val + drift_val * timestep + noise)
        result.append(new_row)
    return result


def euler_step(
    features: List[List[float]],
    adjacency: List[List[float]],
    drift_features: List[List[float]],
    drift_adjacency: List[List[float]],
    timestep: float,
    noise_scale: float,
    rng: random.Random,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Equation 2 / Algorithm 2 lines 15-16: Euler-Maruyama update.

    Args:
        features: Node feature matrix X_{k-1}.
        adjacency: Adjacency matrix A_{k-1}.
        drift_features: Drift f_X evaluated at (X_{k-1}, A_{k-1}, t).
        drift_adjacency: Drift f_A evaluated at (X_{k-1}, A_{k-1}, t).
        timestep: Adapted timestep dt_k.
        noise_scale: g_t * sqrt(dt_k).
        rng: Random number generator for reproducibility.

    Returns:
        The updated pair ``(X_k, A_k)``.
    """
    next_features = _add_drift_and_noise(
        features, drift_features, timestep, noise_scale, rng
    )
    next_adjacency = _add_drift_and_noise(
        adjacency, drift_adjacency, timestep, noise_scale, rng
    )
    return next_features, next_adjacency


def heun_step(
    features: List[List[float]],
    adjacency: List[List[float]],
    first_drift_features: List[List[float]],
    first_drift_adjacency: List[List[float]],
    timestep: float,
    noise_scale: float,
    drift_function: DriftFunction,
    time: float,
    rng: random.Random,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Algorithm 3: Heun predictor-corrector update.

    1. **Predictor** (Euler step using first drift estimate):
       X_hat = X + f1 * dt + g * sqrt(dt) * Z

    2. **Corrector drift** evaluated at the predicted state and t + dt:
       f2 = drift_function(X_hat, A_hat, t + dt)

    3. **Corrector** (trapezoidal average):
       X_k = X + 0.5 * (f1 + f2) * dt + g * sqrt(dt) * Z

    Args:
        features: Node feature matrix X_{k-1}.
        adjacency: Adjacency matrix A_{k-1}.
        first_drift_features: First-stage drift f^{(1)}_X.
        first_drift_adjacency: First-stage drift f^{(1)}_A.
        timestep: Adapted timestep dt_k.
        noise_scale: g_t * sqrt(dt_k).
        drift_function: Callable denoising network drift.
        time: Current diffusion time t.
        rng: Random number generator.

    Returns:
        The corrected pair ``(X_k, A_k)``.
    """
    # Predictor
    predicted_features = _add_drift_and_noise(
        features, first_drift_features, timestep, noise_scale, rng
    )
    predicted_adjacency = _add_drift_and_noise(
        adjacency, first_drift_adjacency, timestep, noise_scale, rng
    )

    # Second drift evaluation at predicted state
    second_drift_features, second_drift_adjacency = drift_function(
        predicted_features, predicted_adjacency, time + timestep
    )

    # Corrector using averaged drift
    def _average_and_update(
        state: List[List[float]],
        drift1: List[List[float]],
        drift2: List[List[float]],
    ) -> List[List[float]]:
        result: List[List[float]] = []
        for state_row, d1_row, d2_row in zip(state, drift1, drift2):
            new_row: List[float] = []
            for s, d1, d2 in zip(state_row, d1_row, d2_row):
                noise = noise_scale * rng.gauss(0.0, 1.0)
                avg_drift = 0.5 * (d1 + d2)
                new_row.append(s + avg_drift * timestep + noise)
            result.append(new_row)
        return result

    next_features = _average_and_update(
        features, first_drift_features, second_drift_features
    )
    next_adjacency = _average_and_update(
        adjacency, first_drift_adjacency, second_drift_adjacency
    )
    return next_features, next_adjacency


class DVSSampler:
    """DVS-driven adaptive sampler (Algorithms 1--3 combined).

    This is a **training-free**, modular component that wraps an existing
    denoising drift function and noise schedule.  It can be plugged into any
    graph diffusion model that exposes a ``drift_function(X, A, t)``
    interface.
    """

    def __init__(
        self,
        drift_function: DriftFunction,
        noise_schedule: NoiseSchedule,
        common_config: CommonConfig,
        dataset_config: DatasetConfig,
        solver: str = "Euler",
        seed: Optional[int] = None,
    ) -> None:
        """Initialise the sampler.

        Args:
            drift_function: Callable ``(X, A, t) -> (f_X, f_A)``.
            noise_schedule: Callable ``t -> g_t``.
            common_config: Common hyperparameters (Table 6).
            dataset_config: Dataset-specific hyperparameters (Table 7).
            solver: One of ``"Euler"`` or ``"Heun"``.
            seed: Optional random seed for reproducibility.

        Raises:
            ValueError: If ``solver`` is not ``"Euler"`` or ``"Heun"``.
            ValueError: If the dataset config does not provide a gamma for the
                requested solver.
        """
        if solver not in {"Euler", "Heun"}:
            raise ValueError(f"solver must be 'Euler' or 'Heun', got {solver!r}")
        self._drift_function = drift_function
        self._noise_schedule = noise_schedule
        self._common_config = common_config
        self._dataset_config = dataset_config
        self._solver = solver
        self._rng = random.Random(seed)

        # Select aggregation factor based on solver (Table 7)
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
        """The aggregation factor used by this sampler instance."""
        # The cast is safe because we validated gamma in __init__.
        return self._gamma  # type: ignore[return-value]

    def sample(
        self,
        initial_features: List[List[float]],
        initial_adjacency: List[List[float]],
        terminal_time: float = 1.0,
        verbose: bool = False,
    ) -> Tuple[List[List[float]], List[List[float]], Dict[str, List[float]]]:
        """Run adaptive sampling from t = 0 to ``terminal_time``.

        Args:
            initial_features: Initial noisy node features X_0.
            initial_adjacency: Initial noisy adjacency A_0.
            terminal_time: Diffusion horizon T (default 1.0).
            verbose: If True, print a line for every solver step.

        Returns:
            A triple ``(X_T, A_T, info_dict)`` where ``info_dict`` contains
            step indices, timesteps, diffusion times, raw DVS values, and
            smoothed DVS values for post-hoc analysis.
        """
        common = self._common_config
        dataset = self._dataset_config

        time = 0.0
        step_index = 1
        smoothed_x = 0.0
        smoothed_a = 0.0

        prev_features = [list(row) for row in initial_features]
        prev_adjacency = [list(row) for row in initial_adjacency]

        # Cache previous drift for DVS computation (Algorithm 2 / 3)
        cached_drift_features: Optional[List[List[float]]] = None
        cached_drift_adjacency: Optional[List[List[float]]] = None

        info: Dict[str, List[float]] = {
            "steps": [],
            "dt": [],
            "time": [],
            "v_x": [],
            "v_a": [],
            "smoothed_x": [],
            "smoothed_a": [],
        }

        while time < terminal_time - common.eps_bound:
            # Evaluate drift at current state
            drift_features, drift_adjacency = self._drift_function(
                prev_features, prev_adjacency, time
            )
            noise_scale_g = self._noise_schedule(time)

            # Determine whether DVS is active at this time
            dvs_active = (
                dataset.is_active(time)
                and step_index > 1
                and cached_drift_features is not None
                and cached_drift_adjacency is not None
            )

            if dvs_active:
                # Mypy narrowing: we checked is not None above.
                assert cached_drift_features is not None
                assert cached_drift_adjacency is not None
                # Equation 13: Drift Variation Score
                v_x, v_a = compute_drift_variation_score(
                    drift_features,
                    cached_drift_features,
                    drift_adjacency,
                    cached_drift_adjacency,
                    noise_scale_g,
                    common.eps_num,
                )
                # Equation 14: EMA smoothing
                smoothed_x, smoothed_a = update_ema(
                    v_x, v_a, smoothed_x, smoothed_a, common.alpha
                )
                # Equation 15: Power-law step-size scaling
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
                # Bottleneck principle: synchronise modalities
                timestep = min(dt_features, dt_adjacency)
                # Mypy narrowing: gamma was validated in __init__.
                assert self._gamma is not None
                # Global variation refresh
                smoothed_x, smoothed_a = global_refresh(
                    smoothed_x, smoothed_a, self._gamma
                )

                if verbose:
                    print(
                        f"step={step_index:4d} time={time:.6f} "
                        f"active dvs dt={timestep:.6f} "
                        f"v_x={v_x:.4e} v_a={v_a:.4e}"
                    )
            else:
                timestep = common.dt_base
                v_x = 0.0
                v_a = 0.0
                if verbose:
                    print(
                        f"step={step_index:4d} time={time:.6f} "
                        f"base   dt={timestep:.6f}"
                    )

            # Clip so we do not overshoot the terminal time
            timestep = min(timestep, terminal_time - time)

            # Solver step
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

            # Record history
            info["steps"].append(float(step_index))
            info["dt"].append(timestep)
            info["time"].append(time)
            info["v_x"].append(v_x)
            info["v_a"].append(v_a)
            info["smoothed_x"].append(smoothed_x)
            info["smoothed_a"].append(smoothed_a)

            # Cache drift for next DVS computation
            cached_drift_features = [list(row) for row in drift_features]
            cached_drift_adjacency = [list(row) for row in drift_adjacency]

            # Advance
            time += timestep
            step_index += 1
            prev_features = next_features
            prev_adjacency = next_adjacency

        info["total_steps"] = [float(step_index - 1)]
        info["final_time"] = [time]
        return prev_features, prev_adjacency, info
