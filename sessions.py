#!/usr/bin/env python3
"""Claude Code sessions — display latest individual sessions in a table."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from usage import (
    SCRIPT_DIR,
    CSV_PATH,
    SETTINGS_PATH,
    load_sessions,
    load_settings,
    find_active_subscription,
    fmt_cost,
    fmt_duration,
    fmt_tokens,
    fmt_number,
)

DEFAULT_LIMIT = 20
MIN_DURATION_MS = 3000  # hide sessions shorter than 3 seconds


# ── Helpers ──────────────────────────────────────────────────────────────────


def fmt_timestamp(ts: datetime, now: datetime, tz: ZoneInfo) -> str:
    """Format timestamp as a relative time string in local timezone.

    Today: '3 minutes ago', '2 hours ago' (via humanize).
    Yesterday: 'yesterday 14:30'.
    Older: 'Feb 8 09:15'.
    """
    import humanize

    local = ts.astimezone(tz)
    local_now = now.astimezone(tz)

    if local.date() == local_now.date():
        return humanize.naturaltime(local_now - local)
    if local.date() == local_now.date() - timedelta(days=1):
        return f"yesterday {local.strftime('%H:%M')}"
    return local.strftime("%b %-d %H:%M")


def ctx_style(pct: float) -> str:
    """Return a rich style string for context percentage thresholds."""
    if pct >= 90:
        return "red"
    if pct >= 70:
        return "yellow"
    return ""


# ── Table rendering ─────────────────────────────────────────────────────────


def render_table(
    df: "pd.DataFrame",  # type: ignore[name-defined]  # noqa: F821
    settings: "Settings",  # type: ignore[name-defined]  # noqa: F821
    color: bool,
    limit: int,
) -> None:
    """Render a sessions table sorted by last_active descending."""
    import pandas as pd
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console(force_terminal=color, no_color=not color, width=200)

    sub = find_active_subscription(
        settings.subscriptions, datetime.now(settings.tz).date()
    )

    now = datetime.now(settings.tz)
    filtered_df = df[df["duration_ms"].fillna(0) >= MIN_DURATION_MS]
    sorted_df = filtered_df.sort_values("last_active", ascending=False)
    total = len(sorted_df)
    if limit > 0 and total > limit:
        sorted_df = sorted_df.head(limit)
        title = f"[bold]Latest Sessions[/]  [dim]({limit} of {total})[/]"
    else:
        title = f"[bold]Latest Sessions[/]  [dim]({total})[/]"

    table = Table(
        title=title,
        title_justify="left",
        show_header=True,
        show_edge=False,
        pad_edge=False,
        box=None,
        padding=(0, 1),
    )
    table.add_column("Last Active", style="dim", no_wrap=True)
    table.add_column("Project", style="cyan", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Cost", style="yellow", justify="right", no_wrap=True)
    table.add_column("Duration", justify="right", no_wrap=True)
    table.add_column("Lines", no_wrap=True)
    table.add_column("Tokens", justify="right", no_wrap=True)
    table.add_column("Ctx", justify="right", no_wrap=True)

    def num(val: object, as_int: bool = False) -> float:
        """Coerce a value to float, treating NaN/None as 0."""
        if val is None or val is pd.NA or val != val:  # NaN != NaN
            return 0
        return int(val) if as_int else float(val)  # type: ignore[arg-type, call-overload]

    for _, row in sorted_df.iterrows():
        # Timestamp
        ts_str = ""
        if pd.notna(row.get("last_active")):
            ts_str = fmt_timestamp(row["last_active"], now, settings.tz)

        # Project
        raw_dir = row.get("project_dir")
        project_name = os.path.basename(str(raw_dir)) if pd.notna(raw_dir) and raw_dir else ""

        # Model
        raw_model = row.get("model_name")
        model_name = str(raw_model) if pd.notna(raw_model) else ""

        # Cost
        cost_str = fmt_cost(num(row.get("cost_usd")), sub)

        # Duration
        dur_str = fmt_duration(num(row.get("duration_ms")))

        # Lines
        added = int(num(row.get("lines_added"), as_int=True))
        removed = int(num(row.get("lines_removed"), as_int=True))
        lines_text = Text()
        lines_text.append(f"+{fmt_number(added)}", style="green")
        lines_text.append(f" -{fmt_number(removed)}", style="red")

        # Tokens
        input_tok = int(num(row.get("total_input_tokens"), as_int=True))
        output_tok = int(num(row.get("total_output_tokens"), as_int=True))
        tokens_str = f"{fmt_tokens(input_tok)}/{fmt_tokens(output_tok)}"

        # Context
        ctx_pct = num(row.get("context_used_pct"))
        ctx_text = Text(f"{ctx_pct:.0f}%", style=ctx_style(ctx_pct))

        table.add_row(
            ts_str, project_name, model_name, cost_str,
            dur_str, lines_text, tokens_str, ctx_text,
        )

    console.print(table)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    dev = "--dev" in sys.argv
    show_all = "--all" in sys.argv

    limit = DEFAULT_LIMIT
    if show_all:
        limit = 0
    elif "--limit" in sys.argv:
        try:
            idx = sys.argv.index("--limit")
            limit = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("sessions: --limit requires an integer argument", file=sys.stderr)
            return 1

    if dev:
        data_dir = os.path.join(SCRIPT_DIR, "data", "dev")
        csv_path = os.path.join(data_dir, "stats.csv")
        settings_path = os.path.join(data_dir, "settings.json")
    else:
        csv_path = CSV_PATH
        settings_path = SETTINGS_PATH

    compact_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "compact.py")]
    if dev:
        compact_cmd.append("--dev")
    subprocess.run(compact_cmd)

    settings = load_settings(settings_path)
    df = load_sessions(csv_path)

    if df.empty:
        print("No session data found.")
        return 0

    color = sys.stdout.isatty()
    render_table(df, settings, color, limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
