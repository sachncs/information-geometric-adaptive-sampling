"""Small stateless helpers: scalar clipping, active-range checks, decoding.

This module collects the minimal set of pure functions that are reused
across the rest of the package.  Keeping them centralised avoids
subtle inconsistencies (e.g. one place using a strict inequality
while another uses inclusive).

Architecture overview
---------------------
* :func:`clip_value` -- scalar clamp with NaN propagation.  Used by
  :func:`~igasgd.sampler.compute_timestep` to enforce
  ``[dt_min, dt_max]`` and by future code that needs to clamp a scalar
  to a closed interval.
* :func:`in_active_range` -- interval membership predicate with the
  convention that an empty list of ranges means "always active".
  Mirrors :meth:`igasgd.config.DatasetConfig.is_active` for callers
  that have a list of tuples rather than a :class:`DatasetConfig`.
* :func:`decode_adjacency` -- threshold binarizer for continuous
  adjacency matrices.
* :func:`sigmoid_decode_adjacency` -- sigmoid + threshold binarizer
  for logit-style adjacency matrices.

Assumptions
-----------
* Adjacency matrices are square ``(N, N)`` for the canonical use
  case, but :func:`decode_adjacency` accepts rectangular matrices
  without modification.
* Inputs are never ``None``; passing ``None`` will raise ``TypeError``
  from the iteration machinery.

Limitations
-----------
* These helpers are intentionally simple -- they are not designed to
  replace a fully-fledged tensor library.  For high-performance
  settings a vectorised NumPy / Torch implementation should be used.
"""

import math


def clip_value(value: float, lower_bound: float, upper_bound: float) -> float:
    """Clamp ``value`` to the closed interval ``[lower_bound, upper_bound]``.

    Used by :func:`~igasgd.sampler.compute_timestep` to enforce the
    admissible timestep interval and by any future code that needs to
    enforce bounded scalars.  NaN inputs are propagated unchanged so
    that numerical failures surface rather than being silently
    masked by clipping.

    Args:
        value: The scalar to clamp.
        lower_bound: Minimum allowed value (inclusive).
        upper_bound: Maximum allowed value (inclusive).

    Returns:
        The clamped value.  If ``value`` is NaN, NaN is returned.

    Edge cases:
        * ``value = NaN`` returns NaN (the IEEE 754 NaN-propagation
          property is preserved).
        * Inverted bounds (``lower > upper``) yield ``lower_bound``
          for any ``value >= upper_bound`` and
          ``max(lower_bound, min(upper_bound, value))`` otherwise.
          Callers should pass valid bounds.

    Complexity:
        O(1).
    """
    if value != value:  # NaN check (NaN is the only float not equal to itself)
        return value
    return max(lower_bound, min(upper_bound, value))


def in_active_range(time: float, ranges: list[tuple[float, float]]) -> bool:
    """Check whether ``time`` falls inside any of the provided intervals.

    This is the canonical predicate for the active-range logic in
    Table 7 of the paper.  An empty list of ranges is treated as
    "always active" so callers can default to a no-op list without
    special casing.

    Args:
        time: Current diffusion time.
        ranges: List of ``(start, end)`` tuples.  Each interval is
            closed on both endpoints.

    Returns:
        ``True`` if ``time`` is inside at least one interval
        (inclusive on both ends), or if ``ranges`` is empty;
        ``False`` otherwise.

    Complexity:
        O(len(ranges)).

    Example:
        >>> in_active_range(0.5, [(0.0, 1.0)])
        True
        >>> in_active_range(1.5, [(0.0, 1.0)])
        False
        >>> in_active_range(0.5, [])
        True
    """
    if not ranges:
        return True
    return any(start <= time <= end for start, end in ranges)


def decode_adjacency(adjacency: list[list[float]], threshold: float = 0.5) -> list[list[float]]:
    """Binarize a continuous adjacency matrix via simple thresholding.

    The paper mentions decoding continuous adjacency to discrete edges
    but does not specify the exact threshold or activation function.
    This implementation uses a configurable threshold, which is the
    most common default in the graph-diffusion literature.  Users may
    replace it with a sigmoid- or softmax-based decoder (see
    :func:`sigmoid_decode_adjacency`) once the exact paper details are
    known.

    Args:
        adjacency: Continuous adjacency matrix of shape ``(N, N)``.
            Rectangular matrices are also accepted.
        threshold: Cutoff above which (or equal to) an entry is
            considered an edge.

    Returns:
        A binarized matrix with entries in ``{0.0, 1.0}``.  The input
        is not modified.

    Edge cases:
        * Values exactly equal to ``threshold`` are decoded to ``1.0``
          (use a slightly higher threshold to make them ``0.0``).
        * Non-square matrices are accepted but the result inherits
          the same shape.

    Complexity:
        O(rows * cols).
    """
    return [[1.0 if value >= threshold else 0.0 for value in row] for row in adjacency]


def sigmoid_decode_adjacency(
    adjacency: list[list[float]], threshold: float = 0.5
) -> list[list[float]]:
    """Binarize an adjacency matrix using a sigmoid activation + threshold.

    Use this decoder when the upstream model produces *logits* rather
    than post-sigmoid probabilities.  The sigmoid is applied
    element-wise before thresholding so the comparison happens in
    ``[0, 1]`` space.

    Args:
        adjacency: Raw adjacency logits or pre-activations.
        threshold: Threshold applied after the sigmoid.

    Returns:
        A binarized adjacency matrix with entries in ``{0.0, 1.0}``.

    Edge cases:
        * ``value = 0`` yields ``sigmoid(0) = 0.5`` which is
          ``>= 0.5`` and therefore decoded to ``1.0`` for the
          default threshold.

    Complexity:
        O(rows * cols) plus the cost of evaluating ``exp``.

    Numerical notes:
        For very large negative inputs the ``math.exp(-value)`` call
        may overflow; callers that expect extreme negative logits
        should pre-clamp or use a numerically stable variant.
    """
    return [
        [1.0 if 1.0 / (1.0 + math.exp(-value)) >= threshold else 0.0 for value in row]
        for row in adjacency
    ]
