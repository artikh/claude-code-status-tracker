#!/usr/bin/env python3
"""Compact session.jsonl into stats.csv — one row per session, latest values only."""

import fcntl
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import IO

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
JSONL_PATH = os.path.join(DATA_DIR, "session.jsonl")
PROCESSING_PATH = os.path.join(DATA_DIR, "session.jsonl.processing")
CSV_PATH = os.path.join(DATA_DIR, "stats.csv")
LOCK_PATH = os.path.join(DATA_DIR, "compact.lock")

CSV_COLUMNS = [
    "session_id",
    "first_active",
    "last_active",
    "model_id",
    "model_name",
    "project_dir",
    "current_dir",
    "transcript_path",
    "claude_code_version",
    "cost_usd",
    "duration_ms",
    "api_duration_ms",
    "lines_added",
    "lines_removed",
    "total_input_tokens",
    "total_output_tokens",
    "context_window_size",
    "context_used_pct",
    "exceeds_200k_tokens",
]


def acquire_lock(lock_path: str, timeout: float = 1.0) -> IO[str] | None:
    """Try to acquire an exclusive lock, polling every 100ms up to timeout.

    Returns the open file object (caller holds lock) or None on timeout.
    """
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    f = open(lock_path, "w")  # noqa: SIM115
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except OSError:
            if time.monotonic() >= deadline:
                f.close()
                return None
            time.sleep(0.1)


def acquire_jsonl() -> str | None:
    """Rename session.jsonl → .processing (or recover existing .processing).

    Returns the path to the file to process, or None if nothing to do.
    """
    if os.path.exists(PROCESSING_PATH):
        return PROCESSING_PATH
    try:
        os.rename(JSONL_PATH, PROCESSING_PATH)
        return PROCESSING_PATH
    except FileNotFoundError:
        return None


def flatten_entry(entry: dict) -> dict:
    """Extract nested JSONL entry fields into a flat dict matching CSV_COLUMNS."""
    data = entry.get("data", {})
    model = data.get("model", {})
    workspace = data.get("workspace", {})
    cost = data.get("cost", {})
    ctx = data.get("context_window", {})

    def v(val: object) -> object:
        return val if val is not None else pd.NA

    return {
        "session_id": entry.get("session_id", ""),
        "first_active": entry.get("timestamp", ""),
        "last_active": entry.get("timestamp", ""),
        "model_id": v(model.get("id")),
        "model_name": v(model.get("display_name")),
        "project_dir": v(workspace.get("project_dir")),
        "current_dir": v(workspace.get("current_dir")),
        "transcript_path": v(data.get("transcript_path")),
        "claude_code_version": v(data.get("version")),
        "cost_usd": v(cost.get("total_cost_usd")),
        "duration_ms": v(cost.get("total_duration_ms")),
        "api_duration_ms": v(cost.get("total_api_duration_ms")),
        "lines_added": v(cost.get("total_lines_added")),
        "lines_removed": v(cost.get("total_lines_removed")),
        "total_input_tokens": v(ctx.get("total_input_tokens")),
        "total_output_tokens": v(ctx.get("total_output_tokens")),
        "context_window_size": v(ctx.get("context_window_size")),
        "context_used_pct": v(ctx.get("used_percentage")),
        "exceeds_200k_tokens": v(data.get("exceeds_200k_tokens")),
    }


def read_jsonl(path: str) -> pd.DataFrame:
    """Parse JSONL file into a DataFrame — one row per session, latest values."""
    rows: list[dict] = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"compact: skipping malformed line {i}", file=sys.stderr)
                continue
            rows.append(flatten_entry(entry))

    if not rows:
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = pd.DataFrame(rows)
    # For each session, track earliest timestamp and keep row with latest timestamp
    first_active = df.groupby("session_id")["first_active"].min().rename("_first")
    df = df.sort_values("last_active").drop_duplicates("session_id", keep="last")
    df = df.set_index("session_id")
    df["first_active"] = first_active
    df = df.reset_index()
    return df[CSV_COLUMNS]


def read_csv(path: str) -> pd.DataFrame:
    """Read existing stats.csv or return an empty DataFrame."""
    if os.path.exists(path):
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame(columns=CSV_COLUMNS)


def merge(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge existing CSV data with new JSONL data — latest values win."""
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    if combined.empty:
        return combined

    # Preserve earliest first_active per session
    first_active = combined.groupby("session_id")["first_active"].min().rename("_first")

    # Keep row with latest last_active per session
    combined = combined.sort_values("last_active").drop_duplicates(
        "session_id", keep="last"
    )
    combined = combined.set_index("session_id")
    combined["first_active"] = first_active
    combined = combined.reset_index()

    # Sort by last_active descending
    combined = combined.sort_values("last_active", ascending=False).reset_index(
        drop=True
    )
    return combined[CSV_COLUMNS]


def write_csv_atomic(path: str, df: pd.DataFrame) -> None:
    """Write DataFrame to CSV atomically via tempfile + rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".csv.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            df.to_csv(f, index=False)
        os.rename(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main() -> int:
    """Compact session.jsonl into stats.csv. Returns 0 on success, 1 on error."""
    lock = acquire_lock(LOCK_PATH)
    if lock is None:
        print("compact: another instance is running, skipping", file=sys.stderr)
        return 0

    try:
        jsonl_path = acquire_jsonl()
        if jsonl_path is None:
            if os.path.exists(CSV_PATH):
                os.utime(CSV_PATH)
            else:
                write_csv_atomic(CSV_PATH, pd.DataFrame(columns=CSV_COLUMNS))
            print("compact: no session.jsonl to process")
            return 0

        new_df = read_jsonl(jsonl_path)
        existing_df = read_csv(CSV_PATH)
        merged = merge(existing_df, new_df)
        write_csv_atomic(CSV_PATH, merged)

        os.unlink(jsonl_path)

        n_new = len(new_df)
        n_total = len(merged)
        print(f"compact: processed {n_new} sessions, {n_total} total in stats.csv")
        return 0
    except Exception as e:
        print(f"compact: error: {e}", file=sys.stderr)
        return 1
    finally:
        lock.close()


if __name__ == "__main__":
    if "--dev" in sys.argv:
        DATA_DIR = os.path.join(SCRIPT_DIR, "data", "dev")
        JSONL_PATH = os.path.join(DATA_DIR, "session.jsonl")
        PROCESSING_PATH = os.path.join(DATA_DIR, "session.jsonl.processing")
        CSV_PATH = os.path.join(DATA_DIR, "stats.csv")
        LOCK_PATH = os.path.join(DATA_DIR, "compact.lock")
    sys.exit(main())
