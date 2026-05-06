"""Noise schedule utilities for the diffusion noise scale g(t).

The paper treats g(t) as locally constant during each solver step because its
variation along the sampling trajectory is orders of magnitude smaller than
the drift variation (see Appendix A.2).  The exact functional form of g(t) is
not specified in the paper, so we provide common parameterizations used in
diffusion models and allow users to supply their own callable.
"""

import math
from typing import Callable

NoiseSchedule = Callable[[float], float]
"""Type alias for a callable noise schedule g(t)."""


class LinearSchedule:
    """Linear interpolation between ``sigma_min`` and ``sigma_max``.

    g(t) = sigma_max * t + sigma_min * (1 - t)
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 1.0) -> None:
        """Initialize the linear schedule.

        Args:
            sigma_min: Noise scale at t = 0.
            sigma_max: Noise scale at t = 1.
        """
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def __call__(self, time: float) -> float:
        """Evaluate the linear noise schedule."""
        return self._sigma_max * time + self._sigma_min * (1.0 - time)


class CosineSchedule:
    """Cosine-based noise schedule.

    This is an approximate implementation of the schedule popularised by
    ``Improved Denoising Diffusion Probabilistic Models`` (Nichol & Dhariwal,
    2021).  Because the paper does not specify which cosine formulation it
    uses, this can be swapped for the exact schedule once known.
    """

    def __init__(self, offset: float = 0.008) -> None:
        """Initialize the cosine schedule.

        Args:
            offset: Small offset preventing division by zero at t = 0.
        """
        self._offset = offset

    def __call__(self, time: float) -> float:
        """Evaluate the approximate cosine noise schedule."""
        return math.tan((time + self._offset) / (1.0 + self._offset) * math.pi / 2.0)


class PolynomialSchedule:
    """Polynomial noise schedule g(t) = sigma_max * t ** exponent + sigma_min.

    This is a flexible family that subsumes linear (exponent=1) and
    square-root-like schedules.
    """

    def __init__(
        self, exponent: float = 1.0, sigma_min: float = 0.01, sigma_max: float = 1.0
    ) -> None:
        """Initialize the polynomial schedule.

        Args:
            exponent: Power-law exponent.
            sigma_min: Minimum noise scale.
            sigma_max: Maximum noise scale.
        """
        self._exponent = exponent
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def __call__(self, time: float) -> float:
        """Evaluate the polynomial noise schedule."""
        powered: float = time**self._exponent
        return self._sigma_max * powered + self._sigma_min


def constant_schedule(value: float = 1.0) -> NoiseSchedule:
    """Return a constant noise schedule useful for testing.

    Args:
        value: The constant noise scale to return for all times.

    Returns:
        A callable that ignores its argument and always returns ``value``.
    """

    def _schedule(_time: float) -> float:
        return value

    return _schedule
