#!/usr/bin/env python3
"""Claude Code usage analytics — display session stats grouped by time period and workspace."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "stats.csv")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")

# ANSI colors (same palette as track-and-status.py)
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

NUMERIC_COLUMNS = [
    "cost_usd",
    "duration_ms",
    "api_duration_ms",
    "lines_added",
    "lines_removed",
    "total_input_tokens",
    "total_output_tokens",
    "context_window_size",
    "context_used_pct",
]

# ── Data layer ───────────────────────────────────────────────────────────────


@dataclass
class Subscription:
    plan: str
    cost: float
    currency: str
    start: date
    end: date
    usd_rate: float = 1.0  # how many USD per 1 unit of currency

    @property
    def cost_usd(self) -> float:
        """Subscription cost converted to USD."""
        return self.cost * self.usd_rate


@dataclass
class Settings:
    tz: ZoneInfo = field(default_factory=lambda: ZoneInfo("UTC"))
    subscriptions: list[Subscription] = field(default_factory=list)


def _parse_subscriptions(raw_list: list[dict]) -> list[Subscription]:
    """Parse subscription dicts. Raises ValueError on invalid entries."""
    subs: list[Subscription] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        currency = str(item.get("currency", "USD"))
        usd_rate = float(item["usd_rate"]) if "usd_rate" in item else (
            1.0 if currency == "USD" else None
        )
        if usd_rate is None:
            raise ValueError(
                f"Subscription '{item.get('plan', '?')}' uses {currency} "
                f"but no usd_rate provided"
            )
        subs.append(Subscription(
            plan=str(item.get("plan", "")),
            cost=float(item["cost"]),
            currency=currency,
            start=date.fromisoformat(item["start"]),
            end=date.fromisoformat(item["end"]),
            usd_rate=usd_rate,
        ))
    return subs


def _check_overlaps(subs: list[Subscription]) -> None:
    """Raise ValueError if any two subscriptions have overlapping date ranges."""
    sorted_subs = sorted(subs, key=lambda s: s.start)
    for i in range(len(sorted_subs) - 1):
        a, b = sorted_subs[i], sorted_subs[i + 1]
        if a.end >= b.start:
            raise ValueError(
                f"Overlapping subscriptions: '{a.plan}' ({a.start}–{a.end}) "
                f"and '{b.plan}' ({b.start}–{b.end})"
            )


def find_active_subscription(
    subs: list[Subscription], today: date
) -> Subscription | None:
    """Return the subscription whose [start, end] contains today, or None."""
    for sub in subs:
        if sub.start <= today <= sub.end:
            return sub
    return None


def load_settings(path: str = SETTINGS_PATH) -> Settings:
    """Load settings from JSON file. Returns defaults on any error."""
    s = Settings()
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return s

    tz_name = raw.get("timezone")
    if isinstance(tz_name, str):
        try:
            s.tz = ZoneInfo(tz_name)
        except (KeyError, ValueError):
            pass

    raw_subs = raw.get("subscriptions", [])
    if isinstance(raw_subs, list):
        try:
            s.subscriptions = _parse_subscriptions(raw_subs)
            _check_overlaps(s.subscriptions)
        except (KeyError, ValueError, TypeError) as e:
            print(f"usage: warning: bad subscriptions config: {e}", file=sys.stderr)
            s.subscriptions = []

    return s


def load_sessions(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load stats.csv, parse timestamps, cast numerics."""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    for col in ("first_active", "last_active"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, format="mixed")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "exceeds_200k_tokens" in df.columns:
        df["exceeds_200k_tokens"] = (
            df["exceeds_200k_tokens"].astype(str).str.lower() == "true"
        )

    return df


# ── Analytics layer ──────────────────────────────────────────────────────────


@dataclass
class PeriodStats:
    session_count: int = 0
    total_cost: float = 0.0
    lines_added: int = 0
    lines_removed: int = 0
    duration_ms: float = 0.0
    api_duration_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_context_used_pct: float = 0.0
    extended_context_count: int = 0

    @property
    def cost_per_line(self) -> float | None:
        total = self.lines_added + self.lines_removed
        if total == 0:
            return None
        return self.total_cost / total

    @property
    def avg_cost_per_session(self) -> float | None:
        if self.session_count == 0:
            return None
        return self.total_cost / self.session_count

    @property
    def api_wait_ratio(self) -> float | None:
        if self.duration_ms == 0:
            return None
        return self.api_duration_ms / self.duration_ms

    @property
    def output_tokens_per_dollar(self) -> float | None:
        if self.total_cost == 0:
            return None
        return self.total_output_tokens / self.total_cost


@dataclass
class TimePeriod:
    name: str
    start: date
    end: date
    stats: PeriodStats


@dataclass
class WorkspaceStats:
    project_dir: str
    stats: PeriodStats
    currency: str = "USD"
    first_active: date | None = None
    last_active: date | None = None


@dataclass
class HourHistogram:
    lines_changed: list[int] = field(default_factory=lambda: [0] * 24)
    cost: list[float] = field(default_factory=lambda: [0.0] * 24)

    @property
    def has_data(self) -> bool:
        return sum(self.lines_changed) > 0 or sum(self.cost) > 0


@dataclass
class UsageReport:
    generated_at: datetime
    tz: ZoneInfo
    time_periods: list[TimePeriod]
    workspaces: list[WorkspaceStats]
    hour_histogram: HourHistogram
    subscription: Subscription | None = None


def compute_stats(df: pd.DataFrame) -> PeriodStats:
    """Compute aggregated stats from a filtered DataFrame."""
    if df.empty:
        return PeriodStats()

    extended = 0
    if "exceeds_200k_tokens" in df.columns:
        extended = int(df["exceeds_200k_tokens"].sum())

    avg_ctx = 0.0
    if "context_used_pct" in df.columns:
        valid = df["context_used_pct"].dropna()
        if len(valid) > 0:
            avg_ctx = float(valid.mean())

    return PeriodStats(
        session_count=len(df),
        total_cost=float(df["cost_usd"].sum()),
        lines_added=int(df["lines_added"].fillna(0).sum()),
        lines_removed=int(df["lines_removed"].fillna(0).sum()),
        duration_ms=float(df["duration_ms"].fillna(0).sum()),
        api_duration_ms=float(df["api_duration_ms"].fillna(0).sum()),
        total_input_tokens=int(df["total_input_tokens"].fillna(0).sum()),
        total_output_tokens=int(df["total_output_tokens"].fillna(0).sum()),
        avg_context_used_pct=avg_ctx,
        extended_context_count=extended,
    )


def filter_by_date_range(
    df: pd.DataFrame, start: date, end: date, tz: ZoneInfo
) -> pd.DataFrame:
    """Filter sessions whose first_active falls within [start, end] in local tz."""
    if df.empty or "first_active" not in df.columns:
        return df
    start_dt = datetime.combine(start, time.min, tzinfo=tz)
    # end is inclusive: include the full day
    end_dt = datetime.combine(end + timedelta(days=1), time.min, tzinfo=tz)
    mask = (df["first_active"] >= start_dt) & (df["first_active"] < end_dt)
    return df[mask]


def compute_time_periods(
    df: pd.DataFrame, tz: ZoneInfo, active_sub: Subscription | None
) -> list[TimePeriod]:
    """Compute stats for Today, Yesterday, This Week, This Month, Billing Period."""
    now = datetime.now(tz)
    today = now.date()
    yesterday = today - timedelta(days=1)

    # Monday-start week
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)

    month_start = today.replace(day=1)
    # Last day of month
    if today.month == 12:
        month_end = today.replace(month=12, day=31)
    else:
        month_end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)

    periods: list[TimePeriod] = [
        TimePeriod(
            name=f"Today ({today.strftime('%b %-d')})",
            start=today,
            end=today,
            stats=compute_stats(filter_by_date_range(df, today, today, tz)),
        ),
        TimePeriod(
            name=f"Yesterday ({yesterday.strftime('%b %-d')})",
            start=yesterday,
            end=yesterday,
            stats=compute_stats(filter_by_date_range(df, yesterday, yesterday, tz)),
        ),
        TimePeriod(
            name=f"This Week ({week_start.strftime('%b %-d')} – {week_end.strftime('%b %-d')})",
            start=week_start,
            end=week_end,
            stats=compute_stats(filter_by_date_range(df, week_start, week_end, tz)),
        ),
        TimePeriod(
            name=f"This Month ({today.strftime('%B')})",
            start=month_start,
            end=month_end,
            stats=compute_stats(filter_by_date_range(df, month_start, month_end, tz)),
        ),
    ]

    if active_sub:
        bp_name = (
            f"Billing Period ({active_sub.start.strftime('%b %-d')}"
            f" – {active_sub.end.strftime('%b %-d')})"
        )
        periods.append(
            TimePeriod(
                name=bp_name,
                start=active_sub.start,
                end=active_sub.end,
                stats=compute_stats(
                    filter_by_date_range(df, active_sub.start, active_sub.end, tz)
                ),
            )
        )

    return periods


def _convert_session_cost(
    cost_usd: float, session_date: date, subscriptions: list[Subscription],
    fallback_rate: float,
) -> float:
    """Convert a single session's USD cost using the appropriate subscription's rate."""
    for sub in subscriptions:
        if sub.start <= session_date <= sub.end:
            return cost_usd / sub.usd_rate
    return cost_usd / fallback_rate


def compute_workspace_stats(
    df: pd.DataFrame, subscriptions: list[Subscription]
) -> list[WorkspaceStats]:
    """Group sessions by project_dir, sorted by cost desc. Skip empty dirs.

    Costs are converted per-session using each subscription's exchange rate,
    then summed. Display currency is the latest subscription's currency.
    """
    if df.empty or "project_dir" not in df.columns:
        return []

    last_sub = max(subscriptions, key=lambda s: s.end) if subscriptions else None
    display_currency = last_sub.currency if last_sub else "USD"
    fallback_rate = last_sub.usd_rate if last_sub else 1.0

    valid = df[df["project_dir"].notna() & (df["project_dir"] != "")]
    if valid.empty:
        return []

    # Convert costs per-session
    converted = valid.copy()
    if subscriptions and "first_active" in converted.columns:
        converted["cost_usd"] = converted.apply(
            lambda row: _convert_session_cost(
                float(row["cost_usd"]),
                row["first_active"].date() if pd.notna(row["first_active"]) else date.min,
                subscriptions,
                fallback_rate,
            ),
            axis=1,
        )

    results: list[WorkspaceStats] = []
    for project_dir, group in converted.groupby("project_dir", sort=False):
        first = last = None
        if "first_active" in group.columns:
            valid_dates = group["first_active"].dropna()
            if len(valid_dates) > 0:
                first = valid_dates.min().date()
                last = valid_dates.max().date()
        results.append(
            WorkspaceStats(
                project_dir=str(project_dir),
                stats=compute_stats(group),
                currency=display_currency,
                first_active=first,
                last_active=last,
            )
        )

    results.sort(key=lambda w: w.stats.total_cost, reverse=True)
    return results


def compute_hour_histogram(df: pd.DataFrame, tz: ZoneInfo) -> HourHistogram:
    """Aggregate lines changed and cost by local hour of session start."""
    hist = HourHistogram()
    if df.empty or "first_active" not in df.columns:
        return hist

    for _, row in df.iterrows():
        ts = row.get("first_active")
        if pd.isna(ts):
            continue
        hour = ts.astimezone(tz).hour
        added = int(row.get("lines_added", 0) or 0)
        removed = int(row.get("lines_removed", 0) or 0)
        hist.lines_changed[hour] += added + removed
        hist.cost[hour] += float(row.get("cost_usd", 0) or 0)

    return hist


def build_report(df: pd.DataFrame, settings: Settings) -> UsageReport:
    """Build the complete usage report from session data and settings."""
    now = datetime.now(settings.tz)
    active_sub = find_active_subscription(settings.subscriptions, now.date())
    return UsageReport(
        generated_at=now,
        tz=settings.tz,
        time_periods=compute_time_periods(df, settings.tz, active_sub),
        workspaces=compute_workspace_stats(df, settings.subscriptions),
        hour_histogram=compute_hour_histogram(df, settings.tz),
        subscription=active_sub,
    )


# ── Rendering layer ─────────────────────────────────────────────────────────

CURRENCY_SYMBOLS: dict[str, str] = {
    "GBP": "£",
    "USD": "$",
    "EUR": "€",
}


def fmt_cost(
    value: float,
    sub: Subscription | None = None,
    currency: str | None = None,
) -> str:
    """Format a cost value.

    If currency is set, value is already in that currency (just format).
    If sub is set, value is in USD and will be converted.
    Otherwise displayed as USD.
    """
    if currency:
        symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")
        v = value
    elif sub and sub.usd_rate > 0:
        v = value / sub.usd_rate
        symbol = CURRENCY_SYMBOLS.get(sub.currency, sub.currency + " ")
    else:
        v = value
        symbol = "$"
    if 0 < v < 0.01:
        return f"{symbol}{v:.4f}"
    return f"{symbol}{v:.2f}"


def fmt_duration(ms: float) -> str:
    """Format milliseconds as human-readable duration."""
    total_seconds = int(ms / 1000)
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes = total_seconds // 60
    if minutes < 60:
        seconds = total_seconds % 60
        if seconds:
            return f"{minutes}m {seconds}s"
        return f"{minutes}m"
    hours = minutes // 60
    remaining_min = minutes % 60
    if remaining_min:
        return f"{hours}h {remaining_min}m"
    return f"{hours}h"


def fmt_tokens(n: int) -> str:
    """Format token counts with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def fmt_number(n: int) -> str:
    """Format integer with thousands separator."""
    return f"{n:,}"


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    import re

    return re.sub(r"\033\[[0-9;]*m", "", text)


def fmt_currency(amount: float, currency: str) -> str:
    """Format amount with currency symbol (e.g. £150.00)."""
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")
    return f"{symbol}{amount:.2f}"


def fmt_sub_pct(
    cost: float, sub: Subscription | None, color: bool,
    pre_converted: bool = False,
) -> str:
    """Format cost as percentage of subscription, with over-budget warning.

    If pre_converted is True, cost is already in the subscription's currency.
    """
    if sub is None or sub.cost <= 0:
        return ""
    if pre_converted:
        pct = cost / sub.cost * 100
    else:
        if sub.cost_usd <= 0:
            return ""
        pct = cost / sub.cost_usd * 100
    r = RED if color else ""
    y = YELLOW if color else ""
    d = DIM if color else ""
    x = RESET if color else ""
    if pct > 100:
        return f" {r}({pct:.0f}% {sub.plan} — OVER BUDGET){x}"
    if pct >= 80:
        return f" {y}({pct:.0f}% {sub.plan}){x}"
    return f" {d}({pct:.0f}% {sub.plan}){x}"


def _render_period_stats(
    s: PeriodStats, color: bool,
    sub: Subscription | None = None,
    currency: str | None = None,
) -> list[str]:
    """Render a PeriodStats block as indented lines.

    Pass sub for USD costs that need conversion.
    Pass currency for pre-converted costs (e.g. workspace stats).
    """
    g = GREEN if color else ""
    r = RED if color else ""
    y = YELLOW if color else ""
    d = DIM if color else ""
    x = RESET if color else ""

    lines: list[str] = []

    sub_pct = fmt_sub_pct(s.total_cost, sub, color, pre_converted=currency is not None)
    lines.append(
        f"  Sessions: {s.session_count}    "
        f"Cost: {y}{fmt_cost(s.total_cost, sub, currency)}{x}{sub_pct}    "
        f"Lines: {g}+{fmt_number(s.lines_added)}{x} / {r}-{fmt_number(s.lines_removed)}{x}"
    )

    dur_str = fmt_duration(s.duration_ms)
    api_str = fmt_duration(s.api_duration_ms)
    ratio_str = ""
    if s.api_wait_ratio is not None:
        ratio_str = f" ({s.api_wait_ratio:.0%})"
    lines.append(f"  Duration: {dur_str}    API wait: {api_str}{ratio_str}")

    parts: list[str] = []
    if s.cost_per_line is not None:
        parts.append(f"Cost/line: {fmt_cost(s.cost_per_line, sub, currency)}")
    parts.append(
        f"Tokens: {fmt_tokens(s.total_input_tokens)} in / {fmt_tokens(s.total_output_tokens)} out"
    )
    lines.append("  " + "    ".join(parts))

    extra: list[str] = []
    if s.avg_context_used_pct > 0:
        extra.append(f"Avg context: {s.avg_context_used_pct:.0f}%")
    if s.extended_context_count > 0:
        extra.append(f"Extended context: {s.extended_context_count}")
    if s.output_tokens_per_dollar is not None:
        extra.append(
            f"Output tokens/$: {fmt_tokens(int(s.output_tokens_per_dollar))}"
        )
    if extra:
        lines.append(f"  {d}{'    '.join(extra)}{x}")

    return lines


def render_terminal(report: UsageReport, color: bool = True) -> str:
    """Render the full usage report for terminal output."""
    c = CYAN if color else ""
    b = BOLD if color else ""
    d = DIM if color else ""
    x = RESET if color else ""

    lines: list[str] = []
    lines.append(f"{b}Claude Code Usage Report{x}")

    tz_name = str(report.tz)
    generated = report.generated_at.strftime("%Y-%m-%d %H:%M")
    sub = report.subscription
    if sub:
        lines.append(
            f"{d}Generated: {generated} {tz_name}    "
            f"Plan: {sub.plan} ({fmt_currency(sub.cost, sub.currency)}){x}"
        )
    else:
        lines.append(f"{d}Generated: {generated} {tz_name}{x}")
    lines.append("\u2550" * 56)
    lines.append("")

    # Time periods
    for period in report.time_periods:
        lines.append(f"{c}\u25b8 {period.name}{x}")
        if period.stats.session_count == 0:
            lines.append(f"  {d}No sessions{x}")
        else:
            lines.extend(_render_period_stats(period.stats, color, sub))
        lines.append("")

    # Workspaces
    if report.workspaces:
        lines.append(f"{b}By Workspace{x}")
        lines.append("\u2500" * 56)
        for ws in report.workspaces:
            date_range = ""
            if ws.first_active and ws.last_active:
                if ws.first_active == ws.last_active:
                    date_range = f"  {d}{ws.first_active.strftime('%b %-d')}{x}"
                else:
                    date_range = (
                        f"  {d}{ws.first_active.strftime('%b %-d')}"
                        f" – {ws.last_active.strftime('%b %-d')}{x}"
                    )
            lines.append(f"  {c}{ws.project_dir}{x}{date_range}")
            lines.extend(
                _render_period_stats(ws.stats, color, currency=ws.currency)
            )
            lines.append("")

    # Hour histograms
    hist = report.hour_histogram
    if hist.has_data:
        tz_name = str(report.tz)
        lines.append(f"{b}Activity by Hour ({tz_name}){x}")
        lines.append("\u2500" * 56)

        max_bar_width = 30

        # Lines changed histogram
        max_lines = max(hist.lines_changed)
        if max_lines > 0:
            lines.append(f"  {d}Lines changed{x}")
            for hour, count in enumerate(hist.lines_changed):
                if count == 0:
                    continue
                bar_width = int(count / max_lines * max_bar_width)
                bar = "\u2588" * bar_width
                lines.append(f"  {hour:02d}:00  {bar} {fmt_number(count)}")
            lines.append("")

        # Cost histogram
        max_cost = max(hist.cost)
        if max_cost > 0:
            lines.append(f"  {d}Cost{x}")
            for hour, amount in enumerate(hist.cost):
                if amount == 0:
                    continue
                bar_width = int(amount / max_cost * max_bar_width)
                bar = "\u2588" * bar_width
                lines.append(
                    f"  {hour:02d}:00  {bar} {fmt_cost(amount, sub)}"
                )
            lines.append("")

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    if "--dev" in sys.argv:
        data_dir = os.path.join(SCRIPT_DIR, "data", "dev")
        csv_path = os.path.join(data_dir, "stats.csv")
        settings_path = os.path.join(data_dir, "settings.json")
    else:
        csv_path = CSV_PATH
        settings_path = SETTINGS_PATH

    settings = load_settings(settings_path)

    today = datetime.now(settings.tz).date()
    active_sub = find_active_subscription(settings.subscriptions, today)
    if active_sub is None:
        print("usage: error: no active subscription found for today", file=sys.stderr)
        if settings.subscriptions:
            latest = max(settings.subscriptions, key=lambda s: s.end)
            print(
                f"  latest subscription '{latest.plan}' ended {latest.end}",
                file=sys.stderr,
            )
        else:
            print("  no subscriptions configured in settings.json", file=sys.stderr)
        return 1

    df = load_sessions(csv_path)

    if df.empty:
        print("No session data found.")
        return 0

    report = build_report(df, settings)
    color = sys.stdout.isatty()
    print(render_terminal(report, color=color))
    return 0


if __name__ == "__main__":
    sys.exit(main())
