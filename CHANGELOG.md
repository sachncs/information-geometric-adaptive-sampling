# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
