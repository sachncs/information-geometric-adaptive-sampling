"""Unit tests for noise schedule implementations."""

import math
import sys

# Ensure the source tree is on the path when running directly.
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src"))

from igasgd import CosineSchedule, LinearSchedule, PolynomialSchedule, constant_schedule


class TestLinearSchedule:
    """Tests for the linear noise schedule."""

    def test_at_zero(self) -> None:
        """Verify at zero."""
        sched = LinearSchedule(sigma_min=0.01, sigma_max=1.0)
        assert abs(sched(0.0) - 0.01) < 1e-12

    def test_at_one(self) -> None:
        """Verify at one."""
        sched = LinearSchedule(sigma_min=0.01, sigma_max=1.0)
        assert abs(sched(1.0) - 1.0) < 1e-12

    def test_at_half(self) -> None:
        """Verify at half."""
        sched = LinearSchedule(sigma_min=0.0, sigma_max=1.0)
        assert abs(sched(0.5) - 0.5) < 1e-12

    def test_decreasing_schedule(self) -> None:
        """Verify decreasing schedule."""
        sched = LinearSchedule(sigma_min=1.0, sigma_max=0.1)
        assert sched(0.0) == 1.0
        assert sched(1.0) == 0.1
        assert sched(0.5) == 0.55

    def test_constant_sigma(self) -> None:
        """Verify constant sigma."""
        sched = LinearSchedule(sigma_min=0.5, sigma_max=0.5)
        assert sched(0.0) == 0.5
        assert sched(0.5) == 0.5
        assert sched(1.0) == 0.5


class TestCosineSchedule:
    """Tests for the cosine-based noise schedule."""

    def test_at_zero_positive(self) -> None:
        """Verify at zero positive."""
        sched = CosineSchedule(offset=0.008)
        val = sched(0.0)
        assert val > 0.0

    def test_at_one(self) -> None:
        """Verify at one."""
        sched = CosineSchedule(offset=0.008)
        val = sched(1.0)
        assert val > 0.0
        assert math.isfinite(val)

    def test_monotonic_increasing(self) -> None:
        """Verify monotonic increasing."""
        sched = CosineSchedule(offset=0.008)
        v0 = sched(0.0)
        v1 = sched(0.5)
        v2 = sched(1.0)
        assert v0 < v1 < v2

    def test_custom_offset(self) -> None:
        """Verify custom offset."""
        sched = CosineSchedule(offset=0.1)
        assert sched(0.0) > 0.0

    def test_evaluates_at_multiple_times(self) -> None:
        """Verify evaluates at multiple times."""
        sched = CosineSchedule(offset=0.008)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = sched(t)
            assert math.isfinite(val)
            assert val > 0.0


class TestPolynomialSchedule:
    """Tests for the polynomial noise schedule."""

    def test_linear_exponent(self) -> None:
        """Verify linear exponent."""
        sched = PolynomialSchedule(exponent=1.0, sigma_min=0.0, sigma_max=1.0)
        assert abs(sched(0.0) - 0.0) < 1e-12
        assert abs(sched(0.5) - 0.5) < 1e-12
        assert abs(sched(1.0) - 1.0) < 1e-12

    def test_quadratic_exponent(self) -> None:
        """Verify quadratic exponent."""
        sched = PolynomialSchedule(exponent=2.0, sigma_min=0.0, sigma_max=1.0)
        assert abs(sched(0.0) - 0.0) < 1e-12
        assert abs(sched(0.5) - 0.25) < 1e-12
        assert abs(sched(1.0) - 1.0) < 1e-12

    def test_square_root_exponent(self) -> None:
        """Verify square root exponent."""
        sched = PolynomialSchedule(exponent=0.5, sigma_min=0.0, sigma_max=1.0)
        assert abs(sched(0.0) - 0.0) < 1e-12
        assert abs(sched(1.0) - 1.0) < 1e-12
        assert abs(sched(0.25) - 0.5) < 1e-12

    def test_sigma_min_offset(self) -> None:
        """Verify sigma min offset."""
        sched = PolynomialSchedule(exponent=1.0, sigma_min=0.01, sigma_max=1.0)
        assert abs(sched(0.0) - 0.01) < 1e-12
        assert abs(sched(1.0) - 1.01) < 1e-12

    def test_zero_exponent(self) -> None:
        """Verify zero exponent."""
        sched = PolynomialSchedule(exponent=0.0, sigma_min=0.0, sigma_max=1.0)
        # 0.0**0.0 == 1.0 in Python, so t=0 evaluates to 1.0
        assert abs(sched(0.0) - 1.0) < 1e-12
        assert abs(sched(0.5) - 1.0) < 1e-12
        assert abs(sched(1.0) - 1.0) < 1e-12


class TestConstantSchedule:
    """Tests for the constant noise schedule factory."""

    def test_default_value(self) -> None:
        """Verify default value."""
        sched = constant_schedule()
        assert sched(0.0) == 1.0
        assert sched(0.5) == 1.0
        assert sched(1.0) == 1.0

    def test_custom_value(self) -> None:
        """Verify custom value."""
        sched = constant_schedule(0.123)
        assert sched(0.0) == 0.123
        assert sched(100.0) == 0.123

    def test_ignores_time_argument(self) -> None:
        """Verify ignores time argument."""
        sched = constant_schedule(2.0)
        assert sched(-1.0) == 2.0
        assert sched(0.0) == 2.0
        assert sched(1.0) == 2.0


if __name__ == "__main__":
    import inspect

    test_classes = [
        TestLinearSchedule,
        TestCosineSchedule,
        TestPolynomialSchedule,
        TestConstantSchedule,
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
