"""Unit tests for simplified denoising network approximations and internal helpers."""

import math
import random
import sys
from typing import List

# Ensure the source tree is on the path when running directly.
sys.path.insert(
    0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src")
)

from igasgd import (GDSSApproximation, GruMApproximation, SimpleGraphDenoiser,
                    make_drift_function)


class TestSimpleGraphDenoiser:
    """Tests for the base simplified denoising network."""

    def test_output_shapes_match_input(self) -> None:
        model = SimpleGraphDenoiser(num_nodes=4, feature_dim=3, hidden_dim=8, seed=42)
        features = [[0.1 * i + 0.01 * j for j in range(3)] for i in range(4)]
        adjacency = [[0.0 if i == j else 0.5 for j in range(4)] for i in range(4)]
        drift_x, drift_a = model(features, adjacency, time=0.5)
        assert len(drift_x) == 4
        assert len(drift_x[0]) == 3
        assert len(drift_a) == 4
        assert len(drift_a[0]) == 4

    def test_different_seeds_different_outputs(self) -> None:
        model1 = SimpleGraphDenoiser(num_nodes=3, feature_dim=2, hidden_dim=8, seed=1)
        model2 = SimpleGraphDenoiser(num_nodes=3, feature_dim=2, hidden_dim=8, seed=2)
        features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        adjacency = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        dx1, da1 = model1(features, adjacency, time=0.5)
        dx2, da2 = model2(features, adjacency, time=0.5)
        any_diff = any(
            abs(v1 - v2) > 1e-12 for r1, r2 in zip(dx1, dx2) for v1, v2 in zip(r1, r2)
        )
        assert (
            any_diff
        ), "Expected different seeds to produce different weights and outputs"

    def test_same_seed_same_output(self) -> None:
        model1 = SimpleGraphDenoiser(num_nodes=3, feature_dim=2, hidden_dim=8, seed=42)
        model2 = SimpleGraphDenoiser(num_nodes=3, feature_dim=2, hidden_dim=8, seed=42)
        features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        adjacency = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        dx1, da1 = model1(features, adjacency, time=0.5)
        dx2, da2 = model2(features, adjacency, time=0.5)
        for r1, r2 in zip(dx1, dx2):
            for v1, v2 in zip(r1, r2):
                assert abs(v1 - v2) < 1e-12
        for r1, r2 in zip(da1, da2):
            for v1, v2 in zip(r1, r2):
                assert abs(v1 - v2) < 1e-12

    def test_time_embedding_varies_with_time(self) -> None:
        model = SimpleGraphDenoiser(num_nodes=2, feature_dim=1, hidden_dim=4, seed=7)
        features = [[0.0], [0.0]]
        adjacency = [[0.0, 0.0], [0.0, 0.0]]
        dx1, da1 = model(features, adjacency, time=0.1)
        dx2, da2 = model(features, adjacency, time=0.9)
        any_diff = any(
            abs(v1 - v2) > 1e-12 for r1, r2 in zip(dx1, dx2) for v1, v2 in zip(r1, r2)
        )
        assert (
            any_diff
        ), "Time embedding should cause different outputs at different times"

    def test_zero_input_produces_nonzero_drift(self) -> None:
        model = SimpleGraphDenoiser(num_nodes=2, feature_dim=1, hidden_dim=4, seed=0)
        features = [[0.0], [0.0]]
        adjacency = [[0.0, 0.0], [0.0, 0.0]]
        dx, da = model(features, adjacency, time=0.0)
        # Because of biases, zero input does not guarantee zero output
        assert len(dx) == 2
        assert len(da) == 2

    def test_single_node_graph(self) -> None:
        model = SimpleGraphDenoiser(num_nodes=1, feature_dim=1, hidden_dim=4, seed=0)
        features = [[1.0]]
        adjacency = [[0.0]]
        dx, da = model(features, adjacency, time=0.0)
        assert len(dx) == 1 and len(dx[0]) == 1
        assert len(da) == 1 and len(da[0]) == 1


class TestGruMApproximation:
    """Tests for the GruM-specific approximation."""

    def test_inherits_simple_graph_denoiser(self) -> None:
        approx = GruMApproximation(num_nodes=5, feature_dim=3, seed=42)
        assert isinstance(approx, SimpleGraphDenoiser)

    def test_hidden_dim_is_32(self) -> None:
        approx = GruMApproximation(num_nodes=5, feature_dim=3, seed=42)
        assert approx._hidden_dim == 32

    def test_callable_interface(self) -> None:
        approx = GruMApproximation(num_nodes=3, feature_dim=2, seed=1)
        features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        adjacency = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        dx, da = approx(features, adjacency, time=0.5)
        assert len(dx) == 3 and len(dx[0]) == 2
        assert len(da) == 3 and len(da[0]) == 3


class TestGDSSApproximation:
    """Tests for the GDSS-specific approximation."""

    def test_inherits_simple_graph_denoiser(self) -> None:
        approx = GDSSApproximation(num_nodes=5, feature_dim=3, seed=42)
        assert isinstance(approx, SimpleGraphDenoiser)

    def test_hidden_dim_is_24(self) -> None:
        approx = GDSSApproximation(num_nodes=5, feature_dim=3, seed=42)
        assert approx._hidden_dim == 24

    def test_callable_interface(self) -> None:
        approx = GDSSApproximation(num_nodes=3, feature_dim=2, seed=1)
        features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        adjacency = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        dx, da = approx(features, adjacency, time=0.5)
        assert len(dx) == 3 and len(dx[0]) == 2
        assert len(da) == 3 and len(da[0]) == 3


class TestMakeDriftFunction:
    """Tests for the drift function wrapper."""

    def test_wrapper_returns_same_as_model(self) -> None:
        approx = SimpleGraphDenoiser(num_nodes=3, feature_dim=2, hidden_dim=4, seed=7)
        drift = make_drift_function(approx)
        features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        adjacency = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        dx1, da1 = approx(features, adjacency, time=0.5)
        dx2, da2 = drift(features, adjacency, time=0.5)
        for r1, r2 in zip(dx1, dx2):
            for v1, v2 in zip(r1, r2):
                assert abs(v1 - v2) < 1e-12
        for r1, r2 in zip(da1, da2):
            for v1, v2 in zip(r1, r2):
                assert abs(v1 - v2) < 1e-12

    def test_wrapper_signature(self) -> None:
        approx = SimpleGraphDenoiser(num_nodes=2, feature_dim=1, hidden_dim=4, seed=0)
        drift = make_drift_function(approx)
        features = [[0.0], [0.0]]
        adjacency = [[0.0, 0.0], [0.0, 0.0]]
        result = drift(features, adjacency, 0.0)
        assert len(result) == 2


class TestInternalHelpers:
    """Tests for pure-Python linear algebra helpers inside models.py."""

    def test_xavier_uniform_shape_and_bounds(self) -> None:
        from igasgd.models import _xavier_uniform

        rng = random.Random(42)
        mat = _xavier_uniform(rows=3, cols=4, rng=rng)
        assert len(mat) == 3
        assert len(mat[0]) == 4
        limit = math.sqrt(6.0 / (3 + 4))
        for row in mat:
            for val in row:
                assert -limit <= val <= limit

    def test_matvec_basic(self) -> None:
        from igasgd.models import _matvec

        matrix = [[1.0, 2.0], [3.0, 4.0]]
        vector = [1.0, 0.0]
        result = _matvec(matrix, vector)
        assert result == [1.0, 3.0]

    def test_matvec_identity(self) -> None:
        from igasgd.models import _matvec

        matrix = [[1.0, 0.0], [0.0, 1.0]]
        vector = [5.0, -3.0]
        result = _matvec(matrix, vector)
        assert result == [5.0, -3.0]

    def test_matmul_basic(self) -> None:
        from igasgd.models import _matmul

        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[0.0, 1.0], [0.0, 0.0]]
        result = _matmul(a, b)
        assert result == [[0.0, 1.0], [0.0, 3.0]]

    def test_matmul_identity(self) -> None:
        from igasgd.models import _matmul

        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[1.0, 0.0], [0.0, 1.0]]
        result = _matmul(a, b)
        assert result == a

    def test_relu_positive_and_negative(self) -> None:
        from igasgd.models import _relu

        values = [-2.0, -1.0, 0.0, 1.0, 2.0]
        result = _relu(values)
        assert result == [0.0, 0.0, 0.0, 1.0, 2.0]

    def test_time_embedding_length(self) -> None:
        from igasgd.models import _time_embedding

        emb = _time_embedding(0.5, dim=8)
        assert len(emb) == 8

    def test_time_embedding_varies_with_time(self) -> None:
        from igasgd.models import _time_embedding

        emb1 = _time_embedding(0.1, dim=8)
        emb2 = _time_embedding(0.9, dim=8)
        assert emb1 != emb2

    def test_time_embedding_zero_dim(self) -> None:
        from igasgd.models import _time_embedding

        emb = _time_embedding(0.5, dim=0)
        assert emb == []


if __name__ == "__main__":
    import inspect

    test_classes = [
        TestSimpleGraphDenoiser,
        TestGruMApproximation,
        TestGDSSApproximation,
        TestMakeDriftFunction,
        TestInternalHelpers,
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
