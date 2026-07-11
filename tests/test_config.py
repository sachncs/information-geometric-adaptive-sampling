"""Unit tests for hyperparameter configuration classes and lookup functions."""

import sys

# Ensure the source tree is on the path when running directly.
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src"))

from igasgd import DATASET_CONFIGS, CommonConfig, DatasetConfig, get_dataset_config


class TestCommonConfig:
    """Tests for the common hyperparameter dataclass (Table 6)."""

    def test_default_values(self) -> None:
        """Verify default values."""
        cfg = CommonConfig()
        assert cfg.alpha == 0.2
        assert cfg.beta == 0.5
        assert cfg.dt_base == 1e-3
        assert cfg.dt_min == 2e-4
        assert cfg.dt_max == 5e-3
        assert cfg.eps_num == 1e-12
        assert cfg.eps_bound == 1e-6

    def test_custom_values(self) -> None:
        """Verify custom values."""
        cfg = CommonConfig(
            alpha=0.5,
            beta=1.0,
            dt_base=0.01,
            dt_min=1e-5,
            dt_max=0.1,
            eps_num=1e-10,
            eps_bound=1e-4,
        )
        assert cfg.alpha == 0.5
        assert cfg.beta == 1.0
        assert cfg.dt_base == 0.01
        assert cfg.dt_min == 1e-5
        assert cfg.dt_max == 0.1
        assert cfg.eps_num == 1e-10
        assert cfg.eps_bound == 1e-4

    def test_frozen_prevents_modification(self) -> None:
        """Verify frozen prevents modification."""
        cfg = CommonConfig()
        try:
            cfg.alpha = 0.9
        except Exception:
            pass
        else:
            raise AssertionError("Expected CommonConfig to be frozen")

    def test_immutability_of_defaults(self) -> None:
        """Verify immutability of defaults."""
        cfg1 = CommonConfig()
        cfg2 = CommonConfig()
        assert cfg1 == cfg2


class TestDatasetConfig:
    """Tests for the dataset-specific hyperparameter dataclass (Table 7)."""

    def test_is_active_full_range(self) -> None:
        """Verify is active full range."""
        cfg = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=[(0.0, 1.0)],
        )
        assert cfg.is_active(0.0)
        assert cfg.is_active(0.5)
        assert cfg.is_active(1.0)

    def test_is_active_partial_range(self) -> None:
        """Verify is active partial range."""
        cfg = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=[(0.3, 0.7)],
        )
        assert not cfg.is_active(0.0)
        assert not cfg.is_active(0.29)
        assert cfg.is_active(0.3)
        assert cfg.is_active(0.5)
        assert cfg.is_active(0.7)
        assert not cfg.is_active(0.71)
        assert not cfg.is_active(1.0)

    def test_is_active_union_range(self) -> None:
        """Verify is active union range."""
        cfg = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=[(0.0, 0.2), (0.8, 1.0)],
        )
        assert cfg.is_active(0.1)
        assert not cfg.is_active(0.5)
        assert cfg.is_active(0.9)

    def test_is_active_empty_range(self) -> None:
        """Verify is active empty range."""
        cfg = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
            active_range=[],
        )
        assert cfg.is_active(-1.0)
        assert cfg.is_active(0.5)
        assert cfg.is_active(100.0)

    def test_gamma_none_allowed(self) -> None:
        """Verify gamma none allowed."""
        cfg = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=None,
            gamma_heun=None,
            active_range=[],
        )
        assert cfg.gamma_euler is None
        assert cfg.gamma_heun is None

    def test_frozen_prevents_modification(self) -> None:
        """Verify frozen prevents modification."""
        cfg = DatasetConfig(
            model="Test",
            dataset="Test",
            kappa_ref=1.0,
            gamma_euler=0.5,
            gamma_heun=0.5,
        )
        try:
            cfg.kappa_ref = 2.0
        except Exception:
            pass
        else:
            raise AssertionError("Expected DatasetConfig to be frozen")


class TestDatasetConfigLookup:
    """Tests for get_dataset_config and the global DATASET_CONFIGS table."""

    def test_all_configs_present(self) -> None:
        """Verify all configs present."""
        expected_keys: list[tuple[str, str]] = [
            ("GruM", "QM9"),
            ("GruM", "ZINC250k"),
            ("GruM", "Planar"),
            ("GruM", "SBM"),
            ("GDSS", "QM9"),
            ("GDSS", "Ego-small"),
            ("GDSS", "Grid"),
            ("GDSS", "Community-small"),
        ]
        for key in expected_keys:
            assert key in DATASET_CONFIGS, f"Missing config for {key}"

    def test_lookup_grum_qm9(self) -> None:
        """Verify lookup grum qm9."""
        cfg = get_dataset_config("GruM", "QM9")
        assert cfg.model == "GruM"
        assert cfg.dataset == "QM9"
        assert cfg.kappa_ref == 1.0
        assert cfg.gamma_euler == 0.22
        assert cfg.gamma_heun == 0.23
        assert cfg.active_range == [(0.0, 1.0)]

    def test_lookup_gdss_qm9(self) -> None:
        """Verify lookup gdss qm9."""
        cfg = get_dataset_config("GDSS", "QM9")
        assert cfg.model == "GDSS"
        assert cfg.dataset == "QM9"
        assert cfg.kappa_ref == 1.0
        assert cfg.gamma_euler == 0.68
        assert cfg.gamma_heun is None
        assert cfg.active_range == [(0.0, 0.2), (0.95, 1.0)]

    def test_lookup_invalid_raises_keyerror(self) -> None:
        """Verify lookup invalid raises keyerror."""
        try:
            get_dataset_config("NonExistent", "NonExistent")
        except KeyError as exc:
            assert "NonExistent" in str(exc)
        else:
            raise AssertionError("Expected KeyError for invalid config lookup")

    def test_config_immutability_via_lookup(self) -> None:
        """Verify config immutability via lookup."""
        cfg = get_dataset_config("GruM", "QM9")
        try:
            cfg.kappa_ref = 999.0
        except Exception:
            pass
        else:
            raise AssertionError("Config from lookup should be frozen")


if __name__ == "__main__":
    import inspect

    test_classes = [
        TestCommonConfig,
        TestDatasetConfig,
        TestDatasetConfigLookup,
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
