#!/usr/bin/env bash
set -euo pipefail

# install.sh — check that prerequisites are on PATH
# Called by `just install` as the first step.

cd "$(dirname "${BASH_SOURCE[0]}")"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RESET='\033[0m'

FAILED=0

pass() { echo -e "  ${GREEN}✔${RESET} $1"; }
fail() { echo -e "  ${RED}✘${RESET} $1"; FAILED=1; }

echo -e "\n${CYAN}Prerequisites${RESET}"

check_cmd() {
    local cmd=$1 hint=$2
    if command -v "$cmd" &>/dev/null; then
        pass "$cmd found"
    else
        fail "$cmd not found — $hint"
    fi
}

check_cmd claude "install from https://docs.anthropic.com/en/docs/claude-code"
check_cmd git    "install via your package manager"
check_cmd just   "install from https://github.com/casey/just"
check_cmd uv     "install from https://docs.astral.sh/uv/"

if command -v python3 &>/dev/null; then
    PY_OK=$(python3 -c "import sys; print(int(sys.version_info >= (3, 13)))" 2>/dev/null || echo 0)
    if [[ "$PY_OK" == "1" ]]; then
        PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
        pass "python3 $PY_VER (>= 3.13)"
    else
        PY_VER=$(python3 --version 2>&1)
        fail "python3 too old ($PY_VER) — need 3.13+"
    fi
else
    fail "python3 not found — install Python 3.13+"
fi

if [[ "$FAILED" -eq 1 ]]; then
    echo -e "\n${RED}Missing prerequisites — install them and re-run.${RESET}"
    exit 1
fi

just install
