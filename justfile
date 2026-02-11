# Run unit tests
test:
    uv run pytest tests/ -v

# Run type checker
typecheck:
    uv run mypy track-and-status.py tests/

# Run all checks
check: typecheck test

# Configure git to use .githooks/ directory
setup:
    git config core.hooksPath .githooks
