# Contributing to igasgd

Thank you for your interest in contributing!

## Code Style

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

Key points:
- Use `snake_case` for functions, methods, and variables.
- Use `PascalCase` for class names.
- Use `UPPER_CASE` for module-level constants.
- Keep line length at or below 100 characters.
- Write docstrings for all public modules, classes, methods, and functions.
- Use type hints everywhere.

## Running Tests

```bash
python -m pytest tests/
```

## Linting

```bash
ruff check src/ tests/
mypy src/igasgd
```

## Pull Request Process

1. Ensure tests pass.
2. Update documentation if needed.
3. Keep changes focused and atomic.
