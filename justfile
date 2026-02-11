# Run unit tests
test:
    uv run pytest tests/ -v

# Compact session log into stats CSV
compact:
    uv run compact.py

# Compact dev session log into dev stats CSV
compact-dev:
    uv run compact.py --dev

# Show usage stats from production data
usage:
    uv run usage.py

# Show usage stats from dev data
usage-dev:
    uv run usage.py --dev

# Run type checker
typecheck:
    uv run mypy track-and-status.py compact.py usage.py tests/

# Run all checks
check: typecheck test

# Configure git to use .githooks/ directory
setup:
    git config core.hooksPath .githooks
