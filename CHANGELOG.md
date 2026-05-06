# Changelog

## [0.1.0] - 2026-05-06

### Added
- Core DVS sampler implementing Algorithms 1, 2, and 3 from the paper.
- Common hyperparameters from Table 6.
- Dataset-specific hyperparameters from Table 7 (GruM and GDSS).
- Euler-Maruyama and Heun solver steps.
- Simplified denoising network approximations for GruM and GDSS.
- Noise schedule utilities (linear, cosine, polynomial).
- Active range support including union intervals.
- Comprehensive test suite with 38 test cases.
- Demo script with CLI interface.
- Fidelity report comparing implementation to paper.
- Reproduction summary and documentation.
