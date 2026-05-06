"""Unit tests for utility helpers: clipping, active ranges, and graph decoding."""

import math
import sys

# Ensure the source tree is on the path when running directly.
sys.path.insert(
    0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src")
)

from igasgd import (clip_value, decode_adjacency, in_active_range,
                    sigmoid_decode_adjacency)


class TestClipValue:
    """Tests for the scalar clipping utility."""

    def test_inside_range(self) -> None:
        assert clip_value(0.5, 0.0, 1.0) == 0.5

    def test_below_range(self) -> None:
        assert clip_value(-0.1, 0.0, 1.0) == 0.0

    def test_above_range(self) -> None:
        assert clip_value(1.5, 0.0, 1.0) == 1.0

    def test_equal_bounds(self) -> None:
        assert clip_value(5.0, 3.0, 3.0) == 3.0

    def test_negative_bounds(self) -> None:
        assert clip_value(-5.0, -10.0, -2.0) == -5.0
        assert clip_value(-15.0, -10.0, -2.0) == -10.0
        assert clip_value(0.0, -10.0, -2.0) == -2.0

    def test_value_equals_lower(self) -> None:
        assert clip_value(0.0, 0.0, 1.0) == 0.0

    def test_value_equals_upper(self) -> None:
        assert clip_value(1.0, 0.0, 1.0) == 1.0

    def test_inverted_bounds_behavior(self) -> None:
        # When lower > upper, max(lower, min(upper, value)) yields lower for any value > upper
        result = clip_value(5.0, 10.0, 1.0)
        assert result == 10.0
        result2 = clip_value(0.5, 10.0, 1.0)
        assert result2 == 10.0


class TestInActiveRange:
    """Tests for the active-range interval check."""

    def test_empty_list_always_active(self) -> None:
        assert in_active_range(0.0, []) is True
        assert in_active_range(100.0, []) is True
        assert in_active_range(-5.0, []) is True

    def test_single_interval_inside(self) -> None:
        assert in_active_range(0.5, [(0.0, 1.0)]) is True

    def test_single_interval_outside(self) -> None:
        assert in_active_range(1.5, [(0.0, 1.0)]) is False

    def test_at_boundaries(self) -> None:
        assert in_active_range(0.0, [(0.0, 1.0)]) is True
        assert in_active_range(1.0, [(0.0, 1.0)]) is True

    def test_union_intervals(self) -> None:
        ranges = [(0.0, 0.2), (0.8, 1.0)]
        assert in_active_range(0.1, ranges) is True
        assert in_active_range(0.9, ranges) is True
        assert in_active_range(0.5, ranges) is False

    def test_multiple_non_overlapping(self) -> None:
        ranges = [(0.0, 0.1), (0.2, 0.3), (0.5, 0.6)]
        assert in_active_range(0.05, ranges) is True
        assert in_active_range(0.25, ranges) is True
        assert in_active_range(0.55, ranges) is True
        assert in_active_range(0.4, ranges) is False

    def test_single_point_interval(self) -> None:
        assert in_active_range(0.5, [(0.5, 0.5)]) is True
        assert in_active_range(0.49, [(0.5, 0.5)]) is False


class TestDecodeAdjacency:
    """Tests for the threshold-based adjacency binarizer."""

    def test_basic_threshold(self) -> None:
        adjacency = [[0.0, 0.6], [0.4, 1.0]]
        decoded = decode_adjacency(adjacency, threshold=0.5)
        assert decoded == [[0.0, 1.0], [0.0, 1.0]]

    def test_exactly_at_threshold(self) -> None:
        adjacency = [[0.5]]
        decoded = decode_adjacency(adjacency, threshold=0.5)
        assert decoded == [[1.0]]

    def test_zero_threshold(self) -> None:
        adjacency = [[0.0, 0.1], [-0.1, 0.0]]
        decoded = decode_adjacency(adjacency, threshold=0.0)
        assert decoded == [[1.0, 1.0], [0.0, 1.0]]

    def test_all_below_threshold(self) -> None:
        adjacency = [[0.1, 0.2], [0.3, 0.4]]
        decoded = decode_adjacency(adjacency, threshold=0.5)
        assert decoded == [[0.0, 0.0], [0.0, 0.0]]

    def test_does_not_modify_input(self) -> None:
        adjacency = [[0.6, 0.4]]
        _ = decode_adjacency(adjacency, threshold=0.5)
        assert adjacency == [[0.6, 0.4]]

    def test_non_square_matrix(self) -> None:
        adjacency = [[0.6, 0.4, 0.8]]
        decoded = decode_adjacency(adjacency, threshold=0.5)
        assert decoded == [[1.0, 0.0, 1.0]]


class TestSigmoidDecodeAdjacency:
    """Tests for the sigmoid-based adjacency binarizer."""

    def test_zero_input_at_threshold(self) -> None:
        adjacency = [[0.0]]
        decoded = sigmoid_decode_adjacency(adjacency, threshold=0.5)
        # sigmoid(0) = 0.5, which is >= 0.5, so it should decode to 1.0
        assert decoded == [[1.0]]

    def test_large_positive_input(self) -> None:
        adjacency = [[10.0]]
        decoded = sigmoid_decode_adjacency(adjacency, threshold=0.5)
        assert decoded == [[1.0]]

    def test_large_negative_input(self) -> None:
        adjacency = [[-10.0]]
        decoded = sigmoid_decode_adjacency(adjacency, threshold=0.5)
        assert decoded == [[0.0]]

    def test_custom_threshold(self) -> None:
        adjacency = [[0.0]]
        decoded = sigmoid_decode_adjacency(adjacency, threshold=0.6)
        # sigmoid(0) = 0.5 < 0.6
        assert decoded == [[0.0]]

    def test_2x2_matrix(self) -> None:
        adjacency = [[0.0, -10.0], [10.0, 0.0]]
        decoded = sigmoid_decode_adjacency(adjacency, threshold=0.5)
        assert decoded == [[1.0, 0.0], [1.0, 1.0]]

    def test_does_not_modify_input(self) -> None:
        adjacency = [[0.0]]
        _ = sigmoid_decode_adjacency(adjacency, threshold=0.5)
        assert adjacency == [[0.0]]


if __name__ == "__main__":
    import inspect

    test_classes = [
        TestClipValue,
        TestInActiveRange,
        TestDecodeAdjacency,
        TestSigmoidDecodeAdjacency,
    ]

    total = 0
    failures = 0
    for cls in test_classes:
        instance = cls()
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if name.startswith("test_"):
                total += 1
                try:
                    method()
                except AssertionError as exc:
                    failures += 1
                    print(f"FAIL: {cls.__name__}.{name} -- {exc}")
                except Exception as exc:
                    failures += 1
                    print(f"ERROR: {cls.__name__}.{name} -- {exc}")

    if failures:
        print(f"\n{failures}/{total} tests failed.")
        raise SystemExit(1)
    else:
        print(f"All {total} tests passed.")
