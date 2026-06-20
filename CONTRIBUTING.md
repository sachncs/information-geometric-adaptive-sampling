# Contributing to igasgd

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branch Naming](#branch-naming)
- [Commit Conventions](#commit-conventions)
- [Coding Standards](#coding-standards)
- [Running Tests](#running-tests)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/igasgd.git
   cd igasgd
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/example/igasgd.git
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature/my-feature
   ```

## Development Setup

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import igasgd; print(igasgd.__version__)"
```

## Branch Naming

Use descriptive branch names with prefixes:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feat/` | New features | `feat/add-numpy-backend` |
| `fix/` | Bug fixes | `fix/ema-reset-edge-case` |
| `docs/` | Documentation | `docs/update-api-reference` |
| `refactor/` | Code restructuring | `refactor/extract-solver-registry` |
| `test/` | Test additions | `test/numerical-stability` |
| `chore/` | Maintenance | `chore/update-ci` |

## Commit Conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/). All commit messages must follow this format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Code style changes (formatting, no logic changes) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks, dependency updates |
| `ci` | CI/CD configuration changes |

### Examples

```
feat: add Numba JIT backend for sampler

fix: handle edge case with single-node graphs in active range check

docs: add API reference for noise schedule classes

test: add numerical stability tests for extreme parameter values

chore: update pytest to v8.0
```

### Scope

Optional scope should be used to identify the affected module:

```
feat(sampler): add adaptive epsilon scheduling
fix(config): handle missing dataset config gracefully
docs(utils): document clip_value NaN behavior
```

## Coding Standards

This project follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

### Key Conventions

- **Line length:** Maximum 100 characters
- **Indentation:** 4 spaces (no tabs)
- **Naming:**
  - `snake_case` for functions, methods, and variables
  - `PascalCase` for class names
  - `UPPER_CASE` for module-level constants
- **Type hints:** Required for all public APIs
- **Docstrings:** Required for all public modules, classes, methods, and functions (Google-style)

### Linting

```bash
# Check for linting errors
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/

# Type check
mypy src/igasgd
```

### Pre-commit Checks

Before submitting a PR, ensure all of the following pass:

```bash
# All checks in one command
ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/igasgd && python -m pytest tests/
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=igasgd --cov-report=term-missing

# Run a specific test file
python -m pytest tests/test_sampler.py

# Run a specific test class
python -m pytest tests/test_sampler.py::TestDVSSamplerEndToEnd

# Run a specific test
python -m pytest tests/test_sampler.py::TestDVSSamplerEndToEnd::test_sample_returns_expected_keys
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_<module>.py`
- Use descriptive class names prefixed with `Test`
- Use descriptive method names prefixed with `test_`
- Each test should test one concept
- Include edge cases and boundary conditions
- Use `pytest` fixtures when appropriate

## Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Run linting and formatting checks
3. Update documentation if you changed public APIs
4. Update `CHANGELOG.md` under the `[Unreleased]` section
5. Keep changes focused and atomic

### PR Guidelines

1. Create a descriptive PR title following commit conventions
2. Fill out the PR template completely
3. Reference related issues (e.g., "Fixes #42")
4. Keep PRs small and focused -- one feature or fix per PR
5. Respond to review feedback promptly
6. Squash commits before merging if requested

### Review Criteria

PRs are reviewed for:

- Correctness: Does it work as intended?
- Tests: Are there adequate tests for the changes?
- Style: Does it follow project coding standards?
- Documentation: Are public APIs documented?
- Performance: Does it introduce any performance regressions?
- Compatibility: Does it maintain backward compatibility?

## Documentation

### API Documentation

- Write docstrings for all public classes, methods, and functions
- Follow Google-style docstring format
- Include type hints for all parameters and return values
- Provide usage examples in docstrings where helpful

### Docstring Example

```python
def compute_drift_variation_score(
    drift_prev: list[list[float]],
    drift_curr: list[list[float]],
    eps_num: float = 1e-12,
) -> float:
    """Compute the Drift Variation Score (DVS) between consecutive steps.

    Implements Equation 13 from the paper. The DVS measures the magnitude
    of change in drift between consecutive Euler-Maruyama steps.

    Args:
        drift_prev: Drift values from the previous step, shape (N, D).
        drift_curr: Drift values from the current step, shape (N, D).
        eps_num: Numerical stability floor to prevent division by zero.

    Returns:
        The scalar DVS value. Higher values indicate larger drift changes.

    Raises:
        ValueError: If input arrays have mismatched dimensions.
    """
```

### Updating Documentation

- Update `docs/` files when adding new features
- Keep README.md examples working and up to date
- Add entries to `CHANGELOG.md` for all user-facing changes
