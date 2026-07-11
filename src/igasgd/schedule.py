"""Noise schedule utilities for the diffusion noise scale ``g(t)``.

The paper treats ``g(t)`` as **locally constant** during each solver
step because its variation along the sampling trajectory is orders of
magnitude smaller than the drift variation (see Appendix A.2 of the
paper).  The exact functional form of ``g(t)`` is not specified, so
this module provides several common parameterisations used in the
diffusion-model literature and lets users supply their own callable.

Architecture overview
---------------------
* :data:`NoiseSchedule` -- type alias for ``Callable[[float], float]``.
* :class:`LinearSchedule` -- linear interpolation between
  ``sigma_min`` and ``sigma_max``.
* :class:`CosineSchedule` -- approximate implementation of the
  cosine schedule popularised by Nichol & Dhariwal (2021).
* :class:`PolynomialSchedule` -- flexible power-law family that
  subsumes linear (``exponent=1``) and square-root-like schedules.
* :func:`constant_schedule` -- factory returning a time-independent
  callable; useful for deterministic tests and debugging.

Why callables instead of objects?
---------------------------------
Storing the schedule as a callable rather than a frozen dataclass keeps
the sampler API simple (``noise_schedule(t)`` is all the consumer ever
needs) and lets users plug in arbitrary schedules (e.g. learned,
piecewise, or schedule-distilled variants) without subclassing.

Assumptions
-----------
* ``t`` is in the closed interval ``[0, 1]`` for the standard
  parameterisations.  Callers may extend this range, but behaviour
  outside ``[0, 1]`` is schedule-specific.
* All schedules return positive reals (the cosine and polynomial
  schedules use a small offset or floor to guarantee positivity).

Limitations
-----------
* The cosine schedule is an *approximation* of the one used in
  Improved DDPM; replace it with the exact formula when known.
* None of the bundled schedules are learned -- they are deterministic
  curves parameterised by a handful of floats.

References:
----------
* Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic
  Models", 2021.
* Ho et al., "Denoising Diffusion Probabilistic Models", 2020.
"""

import math
from collections.abc import Callable

NoiseSchedule = Callable[[float], float]
"""Type alias for a callable noise schedule ``g(t) -> float``."""


class LinearSchedule:
    """Linear interpolation between ``sigma_min`` and ``sigma_max``.

    Implements

        ``g(t) = sigma_max * t + sigma_min * (1 - t)``

    so that ``g(0) = sigma_min`` and ``g(1) = sigma_max``.  The
    schedule is monotonic increasing when ``sigma_max > sigma_min``
    and monotonic decreasing otherwise; equal endpoints collapse the
    schedule to a constant.

    Lifecycle:
        Instances are stateless after construction.  They may be
        reused across many sampling runs without re-initialisation.

    Example:
        >>> sched = LinearSchedule(sigma_min=0.01, sigma_max=0.5)
        >>> sched(0.0), sched(1.0)
        (0.01, 0.5)
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 1.0) -> None:
        """Initialize the linear schedule.

        Args:
            sigma_min: Noise scale at ``t = 0``.
            sigma_max: Noise scale at ``t = 1``.
        """
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def __call__(self, time: float) -> float:
        """Evaluate the linear noise schedule at the given diffusion time.

        Args:
            time: Diffusion time ``t``.  Typically in ``[0, 1]``.

        Returns:
            The interpolated noise scale ``g(t)``.
        """
        return self._sigma_max * time + self._sigma_min * (1.0 - time)


class CosineSchedule:
    """Approximate implementation of the cosine noise schedule.

    This schedule is a simplified analogue of the cosine schedule
    popularised by *Improved Denoising Diffusion Probabilistic Models*
    (Nichol & Dhariwal, 2021):

        ``g(t) = tan((t + offset) / (1 + offset) * pi / 2)``

    The ``offset`` term prevents division-by-zero and the resulting
    singularity at ``t = 0``.  Because the paper does not specify the
    exact cosine formulation it uses, this can be swapped for the
    precise schedule once known.

    Example:
        >>> sched = CosineSchedule(offset=0.008)
        >>> sched(0.0) > 0.0 and math.isfinite(sched(1.0))
        True
    """

    def __init__(self, offset: float = 0.008) -> None:
        """Initialize the cosine schedule.

        Args:
            offset: Small positive offset preventing division by zero
                at ``t = 0``.  The Improved-DDPM paper uses
                ``offset = 0.008``; smaller values move the singularity
                closer to ``t = 0`` but may produce numerical issues.
        """
        self._offset = offset

    def __call__(self, time: float) -> float:
        """Evaluate the approximate cosine noise schedule.

        Args:
            time: Diffusion time ``t`` in ``[0, 1]``.

        Returns:
            The tangent-based noise scale ``g(t)``.  Always positive
            for ``time >= 0`` and finite for ``time <= 1``.

        Edge cases:
            * ``time = 0`` returns ``tan(offset / (1 + offset) * pi / 2)``
              which is strictly positive when ``offset > 0``.
            * ``time = 1`` returns ``tan(pi / 2)`` which is
              mathematically infinite -- callers may want to clamp
              the diffusion horizon slightly below ``1`` for stability.
        """
        return math.tan((time + self._offset) / (1.0 + self._offset) * math.pi / 2.0)


class PolynomialSchedule:
    """Power-law noise schedule ``g(t) = sigma_max * t**exponent + sigma_min``.

    A flexible family of monotonic schedules that subsumes:

    * Linear (``exponent = 1``).
    * Quadratic (``exponent = 2``).
    * Square-root (``exponent = 0.5``).
    * Constant (``exponent = 0`` with ``sigma_min = 0``).

    The schedule is monotonic increasing for positive exponents and
    positive ``sigma_max``.

    Example:
        >>> sched = PolynomialSchedule(exponent=2.0, sigma_min=0.0, sigma_max=1.0)
        >>> sched(0.5)
        0.25
    """

    def __init__(
        self, exponent: float = 1.0, sigma_min: float = 0.01, sigma_max: float = 1.0
    ) -> None:
        """Initialize the polynomial schedule.

        Args:
            exponent: Power-law exponent applied to ``t``.
            sigma_min: Minimum noise scale (floor).
            sigma_max: Maximum noise scale (at ``t = 1``).
        """
        self._exponent = exponent
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def __call__(self, time: float) -> float:
        """Evaluate the polynomial noise schedule.

        Args:
            time: Diffusion time ``t`` in ``[0, 1]``.

        Returns:
            The polynomial noise scale ``sigma_max * t**exponent + sigma_min``.

        Edge cases:
            * ``time = 0`` and ``exponent = 0`` evaluates to ``1.0``
              because Python defines ``0 ** 0 == 1``.
            * Negative ``time`` raises ``ValueError`` (Python's
              behaviour for ``neg ** non-integer``).
        """
        powered: float = time**self._exponent
        return self._sigma_max * powered + self._sigma_min


def constant_schedule(value: float = 1.0) -> NoiseSchedule:
    """Return a constant noise schedule useful for testing.

    The returned callable ignores its argument and always returns
    ``value``.  This is helpful for unit tests that need to isolate
    the behaviour of the sampler from the choice of ``g(t)``.

    Args:
        value: The constant noise scale to return for all times.

    Returns:
        A callable that ignores its argument and always returns
        ``value``.

    Example:
        >>> sched = constant_schedule(0.5)
        >>> sched(0.0), sched(1.0), sched(123.0)
        (0.5, 0.5, 0.5)
    """

    def _schedule(_time: float) -> float:
        return value

    return _schedule
