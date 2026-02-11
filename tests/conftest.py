"""Shared test helpers."""

from __future__ import annotations

import pandas as pd

import usage


def make_sessions_df(
    rows: list[dict],
) -> pd.DataFrame:
    """Build a DataFrame mimicking load_sessions output with sensible defaults."""
    defaults = {
        "session_id": "sess-1",
        "first_active": "2026-02-11T10:00:00+00:00",
        "last_active": "2026-02-11T10:30:00+00:00",
        "model_id": "claude-opus-4-6",
        "model_name": "Opus 4.6",
        "project_dir": "/src/project",
        "current_dir": "/src/project",
        "transcript_path": "/transcripts/sess-1.jsonl",
        "claude_code_version": "2.1.39",
        "cost_usd": 1.0,
        "duration_ms": 600000,
        "api_duration_ms": 300000,
        "lines_added": 100,
        "lines_removed": 20,
        "total_input_tokens": 10000,
        "total_output_tokens": 5000,
        "context_window_size": 200000,
        "context_used_pct": 25.0,
        "exceeds_200k_tokens": False,
    }
    full_rows = []
    for i, row in enumerate(rows):
        r = {**defaults, **row}
        if "session_id" not in row:
            r["session_id"] = f"sess-{i + 1}"
        full_rows.append(r)

    df = pd.DataFrame(full_rows)
    for col in ("first_active", "last_active"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, format="mixed")
    for col in usage.NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
