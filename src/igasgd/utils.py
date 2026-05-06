"""Utility helpers: clipping, active range checks, and graph decoding."""

import math
from typing import List, Tuple


def clip_value(value: float, lower_bound: float, upper_bound: float) -> float:
    """Clamp ``value`` to the closed interval [lower_bound, upper_bound].

    Args:
        value: The scalar to clamp.
        lower_bound: Minimum allowed value.
        upper_bound: Maximum allowed value.

    Returns:
        The clamped value.  If ``value`` is NaN, NaN is returned.
    """
    if value != value:  # NaN check
        return value
    return max(lower_bound, min(upper_bound, value))


def in_active_range(time: float, ranges: List[Tuple[float, float]]) -> bool:
    """Check whether ``time`` falls inside any of the provided intervals.

    Args:
        time: Current diffusion time.
        ranges: List of (start, end) tuples.  An empty list means "always active".

    Returns:
        True if ``time`` is inside at least one interval (inclusive on both ends).
    """
    if not ranges:
        return True
    return any(start <= time <= end for start, end in ranges)


def decode_adjacency(
    adjacency: List[List[float]], threshold: float = 0.5
) -> List[List[float]]:
    """Binarize a continuous adjacency matrix.

    The paper mentions decoding continuous adjacency to discrete edges but does
    not specify the exact threshold or activation function.  This implementation
    uses a simple configurable threshold, which is a common default in graph
    diffusion literature.  Users may replace it with a sigmoid or softmax-based
    decoder once the exact paper details are known.

    Args:
        adjacency: Continuous adjacency matrix of shape (N, N).
        threshold: Cutoff above which an edge is considered present.

    Returns:
        Binarized adjacency matrix with entries in {0.0, 1.0}.
    """
    return [[1.0 if value >= threshold else 0.0 for value in row] for row in adjacency]


def sigmoid_decode_adjacency(
    adjacency: List[List[float]], threshold: float = 0.5
) -> List[List[float]]:
    """Binarize using a sigmoid activation followed by thresholding.

    Args:
        adjacency: Raw adjacency logits or pre-activations.
        threshold: Threshold applied after the sigmoid.

    Returns:
        Binarized adjacency matrix.
    """
    return [
        [1.0 if 1.0 / (1.0 + math.exp(-value)) >= threshold else 0.0 for value in row]
        for row in adjacency
    ]
