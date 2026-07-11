# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each release section lists the underlying git commits in reverse
chronological order, with the full commit id, ISO-8601 datetime
(author local timezone), and the commit subject.

## [Unreleased]

### Added

- Comprehensive module-, class-, function-, and method-level docstrings
  across the entire `igasgd` package (Google style, with "what" and
  "why" rationale).
- One-line docstring on every test method (D102 satisfied across the
  test suite).
- Architecture data-flow diagram in `README.md`.
- `scripts/ci.sh` -- a local runner mirroring `.github/workflows/ci.yml`
  so contributors can verify the workflow before pushing.

### Changed

- README restyled with centred header, shields.io badges, em-dash feature
  bullets, configuration tables for Tables 6 and 7, expanded project
  structure, and the standard Contributing / Code of Conduct / Security /
  License footer sections.
- Migrated `typing.List` / `typing.Tuple` / `typing.Dict` and
  `Optional[T]` annotations to PEP 585 / PEP 604 built-ins
  (`list` / `tuple` / `dict` / `X | None`).
- Switched to `collections.abc.Callable` and added explicit
  `strict=False` to `zip()` calls.
- Reorganised ruff configuration under `[tool.ruff.lint]` to silence
  deprecation warnings.
- Reduced ruff error count from 357 to 0 across `src/`, `tests/`, and
  `examples/`.
- Removed unused imports across all source and test modules.

### Removed

- Deprecation warnings emitted by older ruff configuration layout.

## [0.1.1] - 2026-07-07

### Commits

- `516433aa04fecbfd6fe6a22ffecbe16b63058db5` (`516433a`) — 2026-07-07 02:29:09 +0530 — `sachin` — Merge pull request #1 from sachncs/dependabot/github_actions/actions/setup-python-6
- `c89f084c3957abfd66d26e73f59401547854dde3` (`c89f084`) — 2026-07-07 02:28:55 +0530 — `sachin` — Merge pull request #2 from sachncs/dependabot/github_actions/actions/checkout-7
- `d236da7aa776ca39f1750797beb973381ad2e1e0` (`d236da7`) — 2026-06-20 07:35:00 +0000 — `dependabot[bot]` — ci: bump actions/checkout from 4 to 7
- `29696598184d4fa7f760ee316de7c5ad4a0410ae` (`2969659`) — 2026-06-20 07:34:57 +0000 — `dependabot[bot]` — ci: bump actions/setup-python from 5 to 6
- `67e4c651067d17778d766edcb9aae60a5f39d741` (`67e4c65`) — 2026-06-20 13:04:33 +0530 — `sachin` — version:0.1.1

### Added

- `CODE_OF_CONDUCT.md` with Contributor Covenant v2.1
- `SECURITY.md` with vulnerability reporting guidelines
- `.editorconfig` for consistent editor formatting
- `.gitattributes` for line ending normalization
- `py.typed` PEP 561 marker for type checking support
- `docs/deployment.md` for production deployment guidance
- GitHub issue templates for bug reports and feature requests
- Pull request template
- Dependabot configuration for automated dependency updates
- Comprehensive project documentation in `docs/`

### Changed

- Rewrote `README.md` with full installation, usage, and development instructions
- Expanded `CONTRIBUTING.md` with detailed development workflow
- Updated `CHANGELOG.md` to follow Keep a Changelog format
- Aligned CI workflow with `pyproject.toml` tooling (ruff instead of black/isort)
- Updated `pyproject.toml` with additional project metadata

### Removed

- References to non-existent `FIDELITY_REPORT.md` and `REPRODUCTION_SUMMARY.md`

## [0.1.0] - 2026-05-06

### Commits

- `280152e7fdc2c0927a372abd4ebfcb5359598f49` (`280152e`) — 2026-05-06 09:37:40 +0530 — `sachin` — version:0.1.0

### Added

- Core DVS sampler implementing Algorithms 1, 2, and 3 from the paper
- Common hyperparameters from Table 6 (alpha, beta, dt bounds, epsilons)
- Dataset-specific hyperparameters from Table 7 for GruM and GDSS
- Euler-Maruyama solver step (Algorithm 2)
- Heun solver step (Algorithm 3, predictor-corrector)
- Simplified denoising network approximations for GruM and GDSS
- Noise schedule utilities: linear, cosine, polynomial, and constant
- Active range support including union intervals
- Drift Variation Score computation (Equation 13)
- EMA smoothing for DVS history (Equation 14)
- Power-law timestep scaling (Equation 15)
- Global refresh mechanism for aggregated EMA reset
- Graph adjacency decoding utilities (threshold and sigmoid)
- Value clipping with NaN propagation
- 153 unit tests across 5 test modules
- Demo script with CLI interface (argparse)
- Source distribution with MANIFEST.in