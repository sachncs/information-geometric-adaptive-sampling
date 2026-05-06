"""Unit tests for the DVS sampler core invariants and shape consistency."""

import math
import random
import statistics
import sys
from typing import List, Tuple

# Ensure the source tree is on the path when running directly.
sys.path.insert(
    0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src")
)

from igasgd import (CommonConfig, DatasetConfig, DVSSampler, clip_value,
                    compute_drift_variation_score, compute_timestep,
                    constant_schedule, euler_step, global_refresh, heun_step,
                    update_ema)


def _make_matrix(rows: int, cols: int, fill: float = 1.0) -> List[List[float]]:
    return [[fill for _ in range(cols)] for _ in range(rows)]


def _make_random_matrix(rows: int, cols: int, rng: random.Random) -> List[List[float]]:
    return [[rng.random() for _ in range(cols)] for _ in range(rows)]


class TestDriftVariationScore:
    """Tests for Equation 13: Drift Variation Score."""

    def test_basic_computation(self) -> None:
        f_x0 = [[0.0, 0.0], [0.0, 0.0]]
        f_x1 = [[1.0, 0.0], [0.0, 1.0]]
        f_a0 = [[0.0], [0.0]]
        f_a1 = [[2.0], [2.0]]
        g = 1.0
        eps = 1e-12
        v_x, v_a = compute_drift_variation_score(f_x1, f_x0, f_a1, f_a0, g, eps)
        assert abs(v_x - 2.0) < 1e-9
        assert abs(v_a - 8.0) < 1e-9

    def test_denom_effect(self) -> None:
        f_x0 = [[0.0]]
        f_x1 = [[2.0]]
        f_a0 = [[0.0]]
        f_a1 = [[0.0]]
        g = 2.0
        eps = 1e-12
        v_x, v_a = compute_drift_variation_score(f_x1, f_x0, f_a1, f_a0, g, eps)
        assert abs(v_x - 1.0) < 1e-9  # 4 / 4
        assert v_a == 0.0

    def test_zero_noise_with_epsilon(self) -> None:
        f_x0 = [[0.0]]
        f_x1 = [[1.0]]
        f_a0 = [[0.0]]
        f_a1 = [[0.0]]
        g = 0.0
        eps = 1e-12
        v_x, _ = compute_drift_variation_score(f_x1, f_x0, f_a1, f_a0, g, eps)
        assert abs(v_x - 1e12) < 1e-3  # 1 / 1e-12

    def test_identical_drifts_give_zero(self) -> None:
        f = [[1.0, 2.0], [3.0, 4.0]]
        v_x, v_a = compute_drift_variation_score(f, f, f, f, 1.0, 1e-12)
        assert v_x == 0.0
        assert v_a == 0.0

    def test_large_values(self) -> None:
        f_x0 = [[0.0]]
        f_x1 = [[1e6]]
        f_a0 = [[0.0]]
        f_a1 = [[0.0]]
        g = 1.0
        eps = 1e-12
        v_x, v_a = compute_drift_variation_score(f_x1, f_x0, f_a1, f_a0, g, eps)
        assert abs(v_x - 1e12) / 1e12 < 1e-9  # relative tolerance for large floats
        assert v_a == 0.0

    def test_mismatched_shapes_raises(self) -> None:
        f_x0 = [[0.0, 0.0]]
        f_x1 = [[0.0]]
        f_a0 = [[0.0]]
        f_a1 = [[0.0]]
        try:
            compute_drift_variation_score(f_x1, f_x0, f_a1, f_a0, 1.0, 1e-12)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError on mismatched drift shapes")

    def test_very_small_differences(self) -> None:
        f_x0 = [[1e-8]]
        f_x1 = [[1e-8 + 1e-10]]
        f_a0 = [[0.0]]
        f_a1 = [[0.0]]
        g = 1.0
        eps = 1e-12
        v_x, v_a = compute_drift_variation_score(f_x1, f_x0, f_a1, f_a0, g, eps)
        expected = (1e-10) ** 2  # 1e-20
        assert abs(v_x - expected) < 1e-30
        assert v_a == 0.0


class TestEmaUpdate:
    """Tests for Equation 14: Exponential Moving Average smoothing."""

    def test_initial_update(self) -> None:
        sx, sa = update_ema(1.0, 2.0, 0.0, 0.0, alpha=0.2)
        assert abs(sx - 0.2) < 1e-9
        assert abs(sa - 0.4) < 1e-9

    def test_consecutive_updates(self) -> None:
        sx, sa = update_ema(1.0, 2.0, 0.0, 0.0, alpha=0.2)
        sx, sa = update_ema(1.0, 2.0, sx, sa, alpha=0.2)
        expected = 0.2 * 1.0 + 0.8 * 0.2  # 0.36
        assert abs(sx - expected) < 1e-9

    def test_alpha_one(self) -> None:
        sx, sa = update_ema(5.0, 7.0, 100.0, 200.0, alpha=1.0)
        assert sx == 5.0
        assert sa == 7.0

    def test_alpha_zero(self) -> None:
        sx, sa = update_ema(5.0, 7.0, 100.0, 200.0, alpha=0.0)
        assert sx == 100.0
        assert sa == 200.0

    def test_zero_inputs(self) -> None:
        sx, sa = update_ema(0.0, 0.0, 0.0, 0.0, alpha=0.2)
        assert sx == 0.0
        assert sa == 0.0

    def test_very_small_alpha(self) -> None:
        sx, sa = update_ema(100.0, 200.0, 0.0, 0.0, alpha=1e-6)
        assert abs(sx - 1e-4) < 1e-12
        assert abs(sa - 2e-4) < 1e-12


class TestComputeTimestep:
    """Tests for Equation 15: Power-law step-size scaling."""

    def test_reference_curvature_returns_base(self) -> None:
        dt = compute_timestep(
            smoothed_score=1.0,
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        assert abs(dt - 1e-3) < 1e-12

    def test_high_curvature_clips_to_min(self) -> None:
        dt = compute_timestep(
            smoothed_score=100.0,
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        assert dt < 1e-3
        assert dt >= 2e-4

    def test_low_curvature_clips_to_max(self) -> None:
        dt = compute_timestep(
            smoothed_score=0.01,
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        assert abs(dt - 5e-3) < 1e-12

    def test_beta_two(self) -> None:
        dt = compute_timestep(
            smoothed_score=4.0,
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=2.0,
            eps_num=1e-12,
        )
        # raw_dt = 1e-3 * (0.25)**2 = 6.25e-5, which is below dt_min -> clips to 2e-4
        assert abs(dt - 2e-4) < 1e-15

    def test_very_small_smoothed_score(self) -> None:
        dt = compute_timestep(
            smoothed_score=1e-10,
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        assert abs(dt - 5e-3) < 1e-12

    def test_zero_smoothed_score_with_epsilon(self) -> None:
        dt = compute_timestep(
            smoothed_score=0.0,
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        # ratio = 1.0 / 1e-12 = 1e12, raw_dt = 1e-3 * sqrt(1e12) = 1e3 -> clips to max
        assert abs(dt - 5e-3) < 1e-12

    def test_very_large_smoothed_score(self) -> None:
        dt = compute_timestep(
            smoothed_score=1e12,
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        assert abs(dt - 2e-4) < 1e-12


class TestGlobalRefresh:
    """Tests for the global variation refresh."""

    def test_basic_refresh(self) -> None:
        sx, sa = global_refresh(1.0, 2.0, gamma=0.5)
        assert abs(sx - 1.5) < 1e-9
        assert abs(sa - 1.5) < 1e-9

    def test_zero_gamma(self) -> None:
        sx, sa = global_refresh(10.0, 20.0, gamma=0.0)
        assert sx == 0.0
        assert sa == 0.0

    def test_one_gamma(self) -> None:
        sx, sa = global_refresh(2.0, 4.0, gamma=1.0)
        assert abs(sx - 6.0) < 1e-9
        assert abs(sa - 6.0) < 1e-9

    def test_negative_gamma(self) -> None:
        sx, sa = global_refresh(1.0, 2.0, gamma=-0.5)
        assert abs(sx + 1.5) < 1e-9
        assert abs(sa + 1.5) < 1e-9


class TestEulerStep:
    """Tests for Algorithm 2: Euler-Maruyama update."""

    def test_output_shapes_match_input(self) -> None:
        rng = random.Random(42)
        features = [[0.0, 0.0], [0.0, 0.0]]
        adjacency = [[0.0], [0.0]]
        drift_x = [[1.0, 1.0], [1.0, 1.0]]
        drift_a = [[1.0], [1.0]]
        dt = 0.01
        g = 0.1
        next_x, next_a = euler_step(features, adjacency, drift_x, drift_a, dt, g, rng)
        assert len(next_x) == 2 and len(next_x[0]) == 2
        assert len(next_a) == 2 and len(next_a[0]) == 1

    def test_deterministic_with_zero_noise(self) -> None:
        rng = random.Random(0)
        features = [[1.0, 2.0]]
        adjacency = [[3.0]]
        drift_x = [[0.5, -0.5]]
        drift_a = [[-1.0]]
        dt = 0.1
        g = 0.0  # No noise
        next_x, next_a = euler_step(features, adjacency, drift_x, drift_a, dt, g, rng)
        assert abs(next_x[0][0] - 1.05) < 1e-9
        assert abs(next_x[0][1] - 1.95) < 1e-9
        assert abs(next_a[0][0] - 2.9) < 1e-9

    def test_noise_scale_proportional_to_sqrt_dt(self) -> None:
        rng = random.Random(123)
        features = [[0.0]]
        adjacency = [[0.0]]
        drift_x = [[0.0]]
        drift_a = [[0.0]]
        dt = 0.04
        g = 2.0
        next_x, _ = euler_step(features, adjacency, drift_x, drift_a, dt, g, rng)
        # Noise scale = g * sqrt(dt) = 2 * 0.2 = 0.4
        # Since RNG is deterministic, the value is fixed.
        # We only assert it's non-zero and reasonable.
        assert next_x[0][0] != 0.0
        assert abs(next_x[0][0]) < 2.0  # 99.7% of Gauss(0, 0.4) within this

    def test_zero_dt(self) -> None:
        rng = random.Random(0)
        features = [[1.0, 2.0]]
        adjacency = [[3.0]]
        drift_x = [[0.5, -0.5]]
        drift_a = [[-1.0]]
        dt = 0.0
        g = 0.0
        next_x, next_a = euler_step(features, adjacency, drift_x, drift_a, dt, g, rng)
        assert next_x == features
        assert next_a == adjacency

    def test_large_dt(self) -> None:
        rng = random.Random(0)
        features = [[1.0]]
        adjacency = [[1.0]]
        drift_x = [[10.0]]
        drift_a = [[10.0]]
        dt = 1.0
        g = 0.0
        next_x, next_a = euler_step(features, adjacency, drift_x, drift_a, dt, g, rng)
        assert abs(next_x[0][0] - 11.0) < 1e-9
        assert abs(next_a[0][0] - 11.0) < 1e-9

    def test_different_rng_produces_different_noise(self) -> None:
        rng1 = random.Random(1)
        rng2 = random.Random(2)
        features = [[0.0]]
        adjacency = [[0.0]]
        drift_x = [[0.0]]
        drift_a = [[0.0]]
        dt = 1.0
        g = 1.0
        x1, _ = euler_step(features, adjacency, drift_x, drift_a, dt, g, rng1)
        x2, _ = euler_step(features, adjacency, drift_x, drift_a, dt, g, rng2)
        assert abs(x1[0][0] - x2[0][0]) > 1e-6


class TestHeunStep:
    """Tests for Algorithm 3: Heun predictor-corrector update."""

    def test_output_shapes_match_input(self) -> None:
        rng = random.Random(42)

        def drift_fn(x, a, t):
            return [[1.0, 1.0], [1.0, 1.0]], [[1.0], [1.0]]

        features = [[0.0, 0.0], [0.0, 0.0]]
        adjacency = [[0.0], [0.0]]
        f1_x = [[1.0, 1.0], [1.0, 1.0]]
        f1_a = [[1.0], [1.0]]
        dt = 0.01
        g = 0.1
        next_x, next_a = heun_step(
            features, adjacency, f1_x, f1_a, dt, g, drift_fn, 0.0, rng
        )
        assert len(next_x) == 2 and len(next_x[0]) == 2
        assert len(next_a) == 2 and len(next_a[0]) == 1

    def test_deterministic_with_zero_noise_and_constant_drift(self) -> None:
        rng = random.Random(0)

        def drift_fn(x, a, t):
            return [[1.0]], [[2.0]]

        features = [[0.0]]
        adjacency = [[0.0]]
        f1_x = [[1.0]]
        f1_a = [[2.0]]
        dt = 0.1
        g = 0.0
        next_x, next_a = heun_step(
            features, adjacency, f1_x, f1_a, dt, g, drift_fn, 0.0, rng
        )
        # Heun with constant drift f2 == f1 reduces to Euler
        assert abs(next_x[0][0] - 0.1) < 1e-9
        assert abs(next_a[0][0] - 0.2) < 1e-9

    def test_non_constant_drift(self) -> None:
        rng = random.Random(0)

        def drift_fn(x, a, t):
            # Linear drift: f = -x
            return [[-v for v in row] for row in x], [[-v for v in row] for row in a]

        features = [[1.0]]
        adjacency = [[1.0]]
        f1_x = [[-1.0]]
        f1_a = [[-1.0]]
        dt = 0.1
        g = 0.0
        next_x, next_a = heun_step(
            features, adjacency, f1_x, f1_a, dt, g, drift_fn, 0.0, rng
        )
        # Predictor: x_hat = 1.0 + (-1.0)*0.1 = 0.9
        # f2 = -0.9
        # Corrector: avg = 0.5*(-1.0 + -0.9) = -0.95
        # next = 1.0 + (-0.95)*0.1 = 0.905
        assert abs(next_x[0][0] - 0.905) < 1e-9
        assert abs(next_a[0][0] - 0.905) < 1e-9

    def test_zero_dt(self) -> None:
        rng = random.Random(0)

        def drift_fn(x, a, t):
            return [[1.0]], [[2.0]]

        features = [[1.0]]
        adjacency = [[1.0]]
        f1_x = [[1.0]]
        f1_a = [[2.0]]
        dt = 0.0
        g = 0.0
        next_x, next_a = heun_step(
            features, adjacency, f1_x, f1_a, dt, g, drift_fn, 0.0, rng
        )
        assert next_x == features
        assert next_a == adjacency


class TestDVSSamplerEndToEnd:
    """End-to-end smoke tests for the full sampler."""

    def _make_sampler(self, solver: str = "Euler", active_range=None):
        common = CommonConfig()
        if active_range is None:
            active_range = [(0.0, 1.0)]
        dataset = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=active_range,
        )

        def drift_fn(x, a, t):
            return [[-0.5 * v for v in row] for row in x], [
                [-0.5 * v for v in row] for row in a
            ]

        return DVSSampler(
            drift_function=drift_fn,
            noise_schedule=constant_schedule(0.1),
            common_config=common,
            dataset_config=dataset,
            solver=solver,
            seed=42,
        )

    def test_euler_sampler_runs(self) -> None:
        sampler = self._make_sampler("Euler")
        x0 = [[1.0, 0.0], [0.0, 1.0]]
        a0 = [[0.5], [0.5]]
        x_t, a_t, info = sampler.sample(x0, a0, terminal_time=0.05, verbose=False)
        assert info["total_steps"][0] >= 1.0
        assert info["final_time"][0] >= 0.05 - 1e-6
        assert len(info["dt"]) == int(info["total_steps"][0])

    def test_heun_sampler_runs(self) -> None:
        sampler = self._make_sampler("Heun")
        x0 = [[1.0, 0.0], [0.0, 1.0]]
        a0 = [[0.5], [0.5]]
        x_t, a_t, info = sampler.sample(x0, a0, terminal_time=0.05, verbose=False)
        assert info["total_steps"][0] >= 1.0

    def test_terminal_time_exactly_reached(self) -> None:
        sampler = self._make_sampler("Euler")
        x0 = [[0.0]]
        a0 = [[0.0]]
        x_t, a_t, info = sampler.sample(x0, a0, terminal_time=1.0, verbose=False)
        final_t = info["final_time"][0]
        assert abs(final_t - 1.0) < 1e-5

    def test_boundary_clip_behavior(self) -> None:
        """When close to T, dt must be clipped to T - t."""
        sampler = self._make_sampler("Euler")
        x0 = [[0.0]]
        a0 = [[0.0]]
        terminal = 0.0015
        _, _, info = sampler.sample(x0, a0, terminal_time=terminal, verbose=False)
        dts = info["dt"]
        # Step 1: dt = 0.001 (base), t becomes 0.001
        # Step 2: remaining = 0.0005, so dt must be clipped to 0.0005 < 0.001
        assert any(dt < 1e-3 for dt in dts)
        assert abs(info["final_time"][0] - terminal) < 1e-12

    def test_invalid_solver_raises(self) -> None:
        try:
            self._make_sampler("InvalidSolver")
        except ValueError as exc:
            assert "InvalidSolver" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid solver")

    def test_missing_gamma_raises(self) -> None:
        common = CommonConfig()
        dataset = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=None,
            gamma_heun=None,
            active_range=[(0.0, 1.0)],
        )

        def drift_fn(x, a, t):
            return x, a

        try:
            DVSSampler(
                drift_function=drift_fn,
                noise_schedule=constant_schedule(0.0),
                common_config=common,
                dataset_config=dataset,
                solver="Euler",
                seed=0,
            )
        except ValueError as exc:
            assert "gamma" in str(exc).lower() or "has no gamma" in str(exc)
        else:
            raise AssertionError("Expected ValueError when gamma is missing")

    def test_verbose_mode_does_not_crash(self) -> None:
        import io
        import sys as _sys

        sampler = self._make_sampler("Euler")
        x0 = [[0.0]]
        a0 = [[0.0]]
        old_stdout = _sys.stdout
        _sys.stdout = io.StringIO()
        try:
            sampler.sample(x0, a0, terminal_time=0.005, verbose=True)
        finally:
            _sys.stdout = old_stdout

    def test_inactive_range_uses_base_dt(self) -> None:
        sampler = self._make_sampler("Euler", active_range=[(0.5, 1.0)])
        x0 = [[0.0]]
        a0 = [[0.0]]
        _, _, info = sampler.sample(x0, a0, terminal_time=0.05, verbose=False)
        # Before 0.5, DVS is inactive so all dts should be dt_base
        for dt in info["dt"]:
            assert abs(dt - CommonConfig.dt_base) < 1e-12
        for v_x in info["v_x"]:
            assert v_x == 0.0
        for v_a in info["v_a"]:
            assert v_a == 0.0

    def test_info_dict_has_all_entries(self) -> None:
        sampler = self._make_sampler("Euler")
        x0 = [[1.0]]
        a0 = [[1.0]]
        _, _, info = sampler.sample(x0, a0, terminal_time=0.01, verbose=False)
        assert len(info["steps"]) == len(info["dt"])
        assert len(info["dt"]) == len(info["time"])
        assert len(info["time"]) == len(info["v_x"])
        assert len(info["v_x"]) == len(info["v_a"])
        assert len(info["v_a"]) == len(info["smoothed_x"])
        assert len(info["smoothed_x"]) == len(info["smoothed_a"])
        assert "total_steps" in info
        assert "final_time" in info

    def test_different_seeds_different_trajectories(self) -> None:
        sampler1 = self._make_sampler("Euler")
        sampler2 = DVSSampler(
            drift_function=sampler1._drift_function,
            noise_schedule=sampler1._noise_schedule,
            common_config=sampler1._common_config,
            dataset_config=sampler1._dataset_config,
            solver="Euler",
            seed=999,
        )
        x0 = [[1.0, 2.0]]
        a0 = [[0.5]]
        x1, _, _ = sampler1.sample(x0, a0, terminal_time=0.05, verbose=False)
        x2, _, _ = sampler2.sample(x0, a0, terminal_time=0.05, verbose=False)
        any_diff = any(
            abs(v1 - v2) > 1e-12 for r1, r2 in zip(x1, x2) for v1, v2 in zip(r1, r2)
        )
        assert any_diff


class TestActiveRanges:
    """Tests for the active-range logic used in Table 7."""

    def test_full_range_always_active(self) -> None:
        dataset = DatasetConfig(
            model="GruM",
            dataset="QM9",
            kappa_ref=1.0,
            gamma_euler=0.22,
            gamma_heun=0.23,
            active_range=[(0.0, 1.0)],
        )
        assert dataset.is_active(0.0)
        assert dataset.is_active(0.5)
        assert dataset.is_active(1.0)

    def test_union_range_gdss_qm9(self) -> None:
        """GDSS/QM9 uses [0,0.2] U [0.95,1]."""
        dataset = DatasetConfig(
            model="GDSS",
            dataset="QM9",
            kappa_ref=1.0,
            gamma_euler=0.68,
            active_range=[(0.0, 0.2), (0.95, 1.0)],
        )
        assert dataset.is_active(0.1)
        assert dataset.is_active(0.97)
        assert not dataset.is_active(0.5)
        assert not dataset.is_active(0.9)

    def test_partial_range_planar(self) -> None:
        """GruM/Planar uses [0.5, 1.0]."""
        dataset = DatasetConfig(
            model="GruM",
            dataset="Planar",
            kappa_ref=10.0,
            gamma_euler=0.31,
            gamma_heun=0.30,
            active_range=[(0.5, 1.0)],
        )
        assert not dataset.is_active(0.1)
        assert not dataset.is_active(0.49)
        assert dataset.is_active(0.5)
        assert dataset.is_active(0.75)
        assert dataset.is_active(1.0)

    def test_empty_range_means_always_active(self) -> None:
        dataset = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            active_range=[],
        )
        assert dataset.is_active(-1.0)
        assert dataset.is_active(0.5)
        assert dataset.is_active(2.0)


class TestBottleneckPrinciple:
    """Tests confirming dt = min(dt_X, dt_A)."""

    def test_high_curvature_dominates(self) -> None:
        dt_x = compute_timestep(
            smoothed_score=100.0,
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        dt_a = compute_timestep(
            smoothed_score=0.01,
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        dt_k = min(dt_x, dt_a)
        assert dt_k == dt_x
        assert dt_k < dt_a

    def test_equal_curvature(self) -> None:
        dt_x = compute_timestep(
            smoothed_score=1.0,
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        dt_a = compute_timestep(
            smoothed_score=1.0,
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        dt_k = min(dt_x, dt_a)
        assert dt_k == dt_x == dt_a

    def test_adjacency_high_curvature_dominates(self) -> None:
        dt_x = compute_timestep(
            smoothed_score=0.01,
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        dt_a = compute_timestep(
            smoothed_score=100.0,
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        dt_k = min(dt_x, dt_a)
        assert dt_k == dt_a
        assert dt_k < dt_x


class TestClipValue:
    """Tests for the clipping utility."""

    def test_inside_range(self) -> None:
        assert clip_value(0.5, 0.0, 1.0) == 0.5

    def test_below_range(self) -> None:
        assert clip_value(-0.1, 0.0, 1.0) == 0.0

    def test_above_range(self) -> None:
        assert clip_value(1.5, 0.0, 1.0) == 1.0

    def test_equal_bounds(self) -> None:
        assert clip_value(5.0, 3.0, 3.0) == 3.0


class TestSolverComparison:
    """Compare Euler and Heun on simple deterministic drift."""

    def test_heun_more_accurate_than_euler_on_linear_drift(self) -> None:
        common = CommonConfig()
        dataset = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=[(0.0, 1.0)],
        )

        def drift_fn(x, a, t):
            # Linear drift: f(x, t) = -x  (analytical solution decays exponentially)
            return [[-v for v in row] for row in x], [[-v for v in row] for row in a]

        euler_sampler = DVSSampler(
            drift_function=drift_fn,
            noise_schedule=constant_schedule(0.0),
            common_config=common,
            dataset_config=dataset,
            solver="Euler",
            seed=42,
        )
        heun_sampler = DVSSampler(
            drift_function=drift_fn,
            noise_schedule=constant_schedule(0.0),
            common_config=common,
            dataset_config=dataset,
            solver="Heun",
            seed=42,
        )

        x0 = [[1.0]]
        a0 = [[1.0]]

        x_euler, a_euler, _ = euler_sampler.sample(
            x0, a0, terminal_time=0.1, verbose=False
        )
        x_heun, a_heun, _ = heun_sampler.sample(
            x0, a0, terminal_time=0.1, verbose=False
        )

        # Analytical solution at t=0.1 with dt_base=0.001 (100 steps):
        # x(t) = exp(-t) approx 0.9048
        # Heun (2nd order) should be closer than Euler (1st order)
        analytical = math.exp(-0.1)
        err_euler = abs(x_euler[0][0] - analytical)
        err_heun = abs(x_heun[0][0] - analytical)
        assert err_heun < err_euler


class TestRandomnessAndReproducibility:
    """Ensure that fixing the seed yields deterministic results."""

    def test_same_seed_same_output(self) -> None:
        common = CommonConfig()
        dataset = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=[(0.0, 1.0)],
        )

        def drift_fn(x, a, t):
            return [[-v for v in row] for row in x], [[-v for v in row] for row in a]

        sampler1 = DVSSampler(
            drift_function=drift_fn,
            noise_schedule=constant_schedule(0.1),
            common_config=common,
            dataset_config=dataset,
            solver="Euler",
            seed=123,
        )
        sampler2 = DVSSampler(
            drift_function=drift_fn,
            noise_schedule=constant_schedule(0.1),
            common_config=common,
            dataset_config=dataset,
            solver="Euler",
            seed=123,
        )

        x0 = [[1.0, 2.0], [3.0, 4.0]]
        a0 = [[0.5], [0.5]]

        x1, a1, info1 = sampler1.sample(x0, a0, terminal_time=0.05, verbose=False)
        x2, a2, info2 = sampler2.sample(x0, a0, terminal_time=0.05, verbose=False)

        assert info1["dt"] == info2["dt"]
        assert info1["total_steps"] == info2["total_steps"]
        for r1, r2 in zip(x1, x2):
            for v1, v2 in zip(r1, r2):
                assert abs(v1 - v2) < 1e-12

    def test_different_seed_different_output(self) -> None:
        common = CommonConfig()
        dataset = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=[(0.0, 1.0)],
        )

        def drift_fn(x, a, t):
            return [[-v for v in row] for row in x], [[-v for v in row] for row in a]

        sampler1 = DVSSampler(
            drift_function=drift_fn,
            noise_schedule=constant_schedule(0.1),
            common_config=common,
            dataset_config=dataset,
            solver="Euler",
            seed=123,
        )
        sampler2 = DVSSampler(
            drift_function=drift_fn,
            noise_schedule=constant_schedule(0.1),
            common_config=common,
            dataset_config=dataset,
            solver="Euler",
            seed=456,
        )

        x0 = [[1.0, 2.0], [3.0, 4.0]]
        a0 = [[0.5], [0.5]]

        x1, _, _ = sampler1.sample(x0, a0, terminal_time=0.05, verbose=False)
        x2, _, _ = sampler2.sample(x0, a0, terminal_time=0.05, verbose=False)

        # At least one value should differ (with extremely high probability)
        any_diff = any(
            abs(v1 - v2) > 1e-12 for r1, r2 in zip(x1, x2) for v1, v2 in zip(r1, r2)
        )
        assert any_diff


class TestEdgeCases:
    """Corner cases: empty graphs, single-node graphs, tiny timesteps."""

    def test_single_node_graph(self) -> None:
        common = CommonConfig()
        dataset = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=[(0.0, 1.0)],
        )

        def drift_fn(x, a, t):
            return [[0.0]], [[0.0]]

        sampler = DVSSampler(
            drift_function=drift_fn,
            noise_schedule=constant_schedule(0.0),
            common_config=common,
            dataset_config=dataset,
            solver="Euler",
            seed=0,
        )
        x0 = [[1.0]]
        a0 = [[0.0]]
        x_t, a_t, info = sampler.sample(x0, a0, terminal_time=0.01, verbose=False)
        assert len(x_t) == 1 and len(x_t[0]) == 1
        assert len(a_t) == 1 and len(a_t[0]) == 1

    def test_zero_terminal_time(self) -> None:
        common = CommonConfig()
        dataset = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=[(0.0, 1.0)],
        )

        def drift_fn(x, a, t):
            return x, a

        sampler = DVSSampler(
            drift_function=drift_fn,
            noise_schedule=constant_schedule(0.0),
            common_config=common,
            dataset_config=dataset,
            solver="Euler",
            seed=0,
        )
        x0 = [[1.0]]
        a0 = [[0.0]]
        x_t, a_t, info = sampler.sample(x0, a0, terminal_time=0.0, verbose=False)
        assert info["total_steps"][0] == 0.0
        assert x_t == x0
        assert a_t == a0

    def test_very_small_terminal_time(self) -> None:
        common = CommonConfig()
        dataset = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=[(0.0, 1.0)],
        )

        def drift_fn(x, a, t):
            return x, a

        sampler = DVSSampler(
            drift_function=drift_fn,
            noise_schedule=constant_schedule(0.0),
            common_config=common,
            dataset_config=dataset,
            solver="Euler",
            seed=0,
        )
        x0 = [[1.0]]
        a0 = [[0.0]]
        x_t, a_t, info = sampler.sample(x0, a0, terminal_time=1e-7, verbose=False)
        assert info["total_steps"][0] == 0.0
        assert x_t == x0
        assert a_t == a0

    def test_many_node_graph_runs(self) -> None:
        common = CommonConfig()
        dataset = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=[(0.0, 1.0)],
        )

        def drift_fn(x, a, t):
            return x, a

        sampler = DVSSampler(
            drift_function=drift_fn,
            noise_schedule=constant_schedule(0.0),
            common_config=common,
            dataset_config=dataset,
            solver="Euler",
            seed=0,
        )
        n = 20
        x0 = [[1.0] * 3 for _ in range(n)]
        a0 = [[0.0] * n for _ in range(n)]
        x_t, a_t, info = sampler.sample(x0, a0, terminal_time=0.01, verbose=False)
        assert len(x_t) == n
        assert len(a_t) == n
        assert info["total_steps"][0] >= 1.0


class TestInfoDictStructure:
    """Ensure the info dictionary has the expected keys and types."""

    def test_keys_present(self) -> None:
        common = CommonConfig()
        dataset = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=[(0.0, 1.0)],
        )

        def drift_fn(x, a, t):
            return [[-v for v in row] for row in x], [[-v for v in row] for row in a]

        sampler = DVSSampler(
            drift_function=drift_fn,
            noise_schedule=constant_schedule(0.0),
            common_config=common,
            dataset_config=dataset,
            solver="Euler",
            seed=42,
        )
        x0 = [[1.0]]
        a0 = [[1.0]]
        _, _, info = sampler.sample(x0, a0, terminal_time=0.01, verbose=False)

        expected_keys = {
            "steps",
            "dt",
            "time",
            "v_x",
            "v_a",
            "smoothed_x",
            "smoothed_a",
            "total_steps",
            "final_time",
        }
        assert set(info.keys()) == expected_keys
        assert all(isinstance(v, list) for v in info.values())

    def test_total_steps_is_scalar_list(self) -> None:
        common = CommonConfig()
        dataset = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=[(0.0, 1.0)],
        )

        def drift_fn(x, a, t):
            return x, a

        sampler = DVSSampler(
            drift_function=drift_fn,
            noise_schedule=constant_schedule(0.0),
            common_config=common,
            dataset_config=dataset,
            solver="Euler",
            seed=0,
        )
        _, _, info = sampler.sample([[0.0]], [[0.0]], terminal_time=0.01, verbose=False)
        assert len(info["total_steps"]) == 1
        assert len(info["final_time"]) == 1
        assert isinstance(info["total_steps"][0], float)
        assert isinstance(info["final_time"][0], float)


class TestNumericalStability:
    """Tests for numerical edge cases in the sampler core."""

    def test_dvs_with_infinity_drift(self) -> None:
        f_x0 = [[0.0]]
        f_x1 = [[float("inf")]]
        f_a0 = [[0.0]]
        f_a1 = [[0.0]]
        v_x, v_a = compute_drift_variation_score(f_x1, f_x0, f_a1, f_a0, 1.0, 1e-12)
        assert v_x == float("inf")
        assert v_a == 0.0

    def test_dvs_with_nan_drift(self) -> None:
        f_x0 = [[0.0]]
        f_x1 = [[float("nan")]]
        f_a0 = [[0.0]]
        f_a1 = [[0.0]]
        v_x, v_a = compute_drift_variation_score(f_x1, f_x0, f_a1, f_a0, 1.0, 1e-12)
        assert math.isnan(v_x)
        assert v_a == 0.0

    def test_ema_with_infinity(self) -> None:
        sx, sa = update_ema(float("inf"), 1.0, 0.0, 0.0, alpha=0.2)
        assert sx == float("inf")
        assert abs(sa - 0.2) < 1e-12

    def test_timestep_with_infinity_score(self) -> None:
        dt = compute_timestep(
            smoothed_score=float("inf"),
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        assert dt == 2e-4

    def test_timestep_with_nan_score(self) -> None:
        dt = compute_timestep(
            smoothed_score=float("nan"),
            kappa_ref=1.0,
            dt_base=1e-3,
            dt_min=2e-4,
            dt_max=5e-3,
            beta=0.5,
            eps_num=1e-12,
        )
        assert math.isnan(dt)

    def test_euler_with_infinity_drift(self) -> None:
        rng = random.Random(0)
        features = [[0.0]]
        adjacency = [[0.0]]
        drift_x = [[float("inf")]]
        drift_a = [[0.0]]
        dt = 0.1
        g = 0.0
        next_x, next_a = euler_step(features, adjacency, drift_x, drift_a, dt, g, rng)
        assert next_x[0][0] == float("inf")
        assert next_a[0][0] == 0.0


if __name__ == "__main__":
    # Run all test classes manually when executed directly.
    import inspect

    test_classes = [
        TestDriftVariationScore,
        TestEmaUpdate,
        TestComputeTimestep,
        TestGlobalRefresh,
        TestEulerStep,
        TestHeunStep,
        TestDVSSamplerEndToEnd,
        TestActiveRanges,
        TestBottleneckPrinciple,
        TestClipValue,
        TestSolverComparison,
        TestRandomnessAndReproducibility,
        TestEdgeCases,
        TestInfoDictStructure,
        TestNumericalStability,
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
