#!/usr/bin/env python3
"""Claude Code status line: context bar, line stats, project stage, and cost."""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

# ANSI colors
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"

CACHE_FILE = "/tmp/claudecode-stage-cache"
CACHE_MAX_AGE = 30  # seconds

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "data")
LOG_FILE = os.path.join(LOG_DIR, "session.jsonl")


def log_event(data: dict, session_id: str) -> str | None:
    """Append a JSON line to the session log. Returns error string on failure."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "data": data,
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        return str(e)
    return None


def get_stage(cwd: str) -> str:
    """Extract current stage from CLAUDE.md, cached for 30s."""
    stale = True
    if os.path.exists(CACHE_FILE):
        age = time.time() - os.path.getmtime(CACHE_FILE)
        stale = age > CACHE_MAX_AGE

    if stale:
        claude_md = os.path.join(cwd, "CLAUDE.md")
        stage = ""
        if os.path.isfile(claude_md):
            with open(claude_md) as f:
                found_section = False
                for line in f:
                    if line.startswith("## Current Stage"):
                        found_section = True
                        continue
                    if found_section and re.match(r"^Stage \d", line):
                        stage = line.strip()
                        break
        with open(CACHE_FILE, "w") as f:
            f.write(stage)
        return stage

    with open(CACHE_FILE) as f:
        return f.read().strip()


def build_bar(pct: int, width: int = 10) -> tuple[str, str]:
    """Build a progress bar and pick its color based on thresholds."""
    if pct >= 90:
        color = RED
    elif pct >= 70:
        color = YELLOW
    else:
        color = GREEN
    filled = pct * width // 100
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return bar, color


def main() -> None:
    data = json.load(sys.stdin)

    model = data.get("model", {}).get("display_name", "?")
    pct = int(data.get("context_window", {}).get("used_percentage") or 0)
    cost = float(data.get("cost", {}).get("total_cost_usd") or 0)
    lines_add = int(data.get("cost", {}).get("total_lines_added") or 0)
    lines_del = int(data.get("cost", {}).get("total_lines_removed") or 0)
    cwd = data.get("workspace", {}).get("current_dir", "")
    session_id = data.get("session_id", "")

    log_err = log_event(data, session_id)

    bar, bar_color = build_bar(pct)
    stage = get_stage(cwd)

    parts = [
        f"{CYAN}[{model}]{RESET}",
        f"{bar_color}{bar}{RESET} {pct}%",
        f"{GREEN}+{lines_add}{RESET} {RED}-{lines_del}{RESET} lines",
    ]
    if stage:
        parts.append(stage)
    parts.append(f"{YELLOW}${cost:.2f}{RESET}")
    if log_err:
        parts.append(f"{RED}[LOG ERR]{RESET}")

    sep = f" {DIM}|{RESET} "
    print(sep.join(parts))


if __name__ == "__main__":
    main()
