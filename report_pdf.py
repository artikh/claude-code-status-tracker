#!/usr/bin/env python3
"""Generate PDF usage reports for finished Claude Code subscriptions."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from fpdf import FPDF

from usage import (
    CURRENCY_SYMBOLS,
    HourHistogram,
    PeriodStats,
    Settings,
    Subscription,
    WorkspaceStats,
    compute_hour_histogram,
    compute_stats,
    compute_workspace_stats,
    filter_by_date_range,
    fmt_cost,
    fmt_currency,
    fmt_duration,
    fmt_number,
    fmt_sub_pct,
    fmt_tokens,
    load_sessions,
    load_settings,
)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "stats.csv")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")


# ── Data model ──────────────────────────────────────────────────────────────


@dataclass
class SubscriptionReport:
    subscription: Subscription
    stats: PeriodStats
    workspaces: list[WorkspaceStats]
    hour_histogram: HourHistogram


# ── Analytics ───────────────────────────────────────────────────────────────


def find_finished_subscriptions(
    subs: list[Subscription], today: date
) -> list[Subscription]:
    """Return subscriptions where end < today, sorted by start ascending."""
    return sorted(
        [s for s in subs if s.end < today],
        key=lambda s: s.start,
    )


def build_subscription_report(
    df: pd.DataFrame, sub: Subscription, tz: ZoneInfo
) -> SubscriptionReport:
    """Build a report for a single subscription period."""
    filtered = filter_by_date_range(df, sub.start, sub.end, tz)
    return SubscriptionReport(
        subscription=sub,
        stats=compute_stats(filtered),
        workspaces=compute_workspace_stats(filtered, [sub]),
        hour_histogram=compute_hour_histogram(filtered, tz),
    )


# ── PDF renderer ────────────────────────────────────────────────────────────


# Colors (R, G, B)
COLOR_BLUE = (66, 133, 244)
COLOR_GOLD = (212, 175, 55)
COLOR_GREEN = (52, 168, 83)
COLOR_RED = (234, 67, 53)
COLOR_GRAY = (128, 128, 128)
COLOR_BLACK = (0, 0, 0)
COLOR_DARK = (51, 51, 51)


class PdfReport:
    def __init__(self, tz: ZoneInfo) -> None:
        self.tz = tz
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=20)

    def generate(
        self, reports: list[SubscriptionReport], settings: Settings
    ) -> None:
        self._title_page(reports, settings)
        for report in reports:
            self._subscription_page(report)

    def save(self, path: str) -> None:
        self.pdf.output(path)

    # ── Title page ──────────────────────────────────────────────────────

    def _title_page(
        self, reports: list[SubscriptionReport], settings: Settings
    ) -> None:
        pdf = self.pdf
        pdf.add_page()

        # Title
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 15, "Claude Code Usage Report", new_x="LMARGIN", new_y="NEXT")

        # Generation timestamp
        now = datetime.now(self.tz)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*COLOR_GRAY)
        pdf.cell(
            0, 7,
            f"Generated: {now.strftime('%Y-%m-%d %H:%M')} {self.tz}",
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_text_color(*COLOR_BLACK)
        pdf.ln(10)

        # Summary table
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Finished Subscriptions", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        col_widths = [35, 42, 30, 28, 20, 20]
        headers = ["Plan", "Period", "Budget", "Used", "%", "Sessions"]

        # Table header
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(240, 240, 240)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 8, header, border=1, fill=True)
        pdf.ln()

        # Table rows
        pdf.set_font("Helvetica", "", 9)
        for r in reports:
            sub = r.subscription
            period = f"{sub.start.strftime('%b %-d')} - {sub.end.strftime('%b %-d, %Y')}"
            budget = fmt_currency(sub.cost, sub.currency)
            used = fmt_cost(r.stats.total_cost, sub)
            if sub.cost_usd > 0:
                pct = f"{r.stats.total_cost / sub.cost_usd * 100:.0f}%"
            else:
                pct = "-"
            sessions = str(r.stats.session_count)

            cells = [sub.plan, period, budget, used, pct, sessions]
            for i, cell_text in enumerate(cells):
                pdf.cell(col_widths[i], 7, cell_text, border=1)
            pdf.ln()

    # ── Per-subscription page ───────────────────────────────────────────

    def _subscription_page(self, report: SubscriptionReport) -> None:
        pdf = self.pdf
        sub = report.subscription
        stats = report.stats

        pdf.add_page()

        # Header
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(*COLOR_BLACK)
        pdf.cell(0, 12, sub.plan, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*COLOR_GRAY)
        period = (
            f"{sub.start.strftime('%b %-d, %Y')} - {sub.end.strftime('%b %-d, %Y')}"
            f"    Budget: {fmt_currency(sub.cost, sub.currency)}"
        )
        pdf.cell(0, 7, period, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*COLOR_BLACK)
        pdf.ln(6)

        # Overall stats
        self._stats_block(stats, sub)
        pdf.ln(4)

        # Workspace table
        if report.workspaces:
            self._workspace_table(report.workspaces, sub)
            pdf.ln(4)

        # Hour histograms
        if report.hour_histogram.has_data:
            self._hour_histograms(report.hour_histogram, sub)

    def _stats_block(self, stats: PeriodStats, sub: Subscription) -> None:
        pdf = self.pdf
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Overview", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # Build stat pairs
        cost_str = fmt_cost(stats.total_cost, sub)
        sub_pct = fmt_sub_pct(stats.total_cost, sub, color=False)
        rows = [
            ("Sessions", str(stats.session_count)),
            ("Cost", f"{cost_str}{sub_pct}"),
            ("Lines", f"+{fmt_number(stats.lines_added)} / -{fmt_number(stats.lines_removed)}"),
            ("Duration", fmt_duration(stats.duration_ms)),
            ("API wait", f"{fmt_duration(stats.api_duration_ms)}"
                + (f" ({stats.api_wait_ratio:.0%})" if stats.api_wait_ratio is not None else "")),
        ]
        if stats.cost_per_line is not None:
            rows.append(("Cost/line", fmt_cost(stats.cost_per_line, sub)))
        rows.append((
            "Tokens",
            f"{fmt_tokens(stats.total_input_tokens)} in / {fmt_tokens(stats.total_output_tokens)} out",
        ))
        if stats.avg_context_used_pct > 0:
            rows.append(("Avg context", f"{stats.avg_context_used_pct:.0f}%"))
        if stats.extended_context_count > 0:
            rows.append(("Extended context", str(stats.extended_context_count)))

        pdf.set_font("Helvetica", "", 10)
        for label, value in rows:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(35, 6, label)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

    def _workspace_table(
        self, workspaces: list[WorkspaceStats], sub: Subscription
    ) -> None:
        pdf = self.pdf

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Workspaces", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        col_widths = [52, 20, 25, 22, 22, 25, 25]
        headers = ["Path", "Sessions", "Cost", "+Lines", "-Lines", "Duration", "Dates"]

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(240, 240, 240)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 7, header, border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for ws in workspaces:
            # Truncate long paths
            path = ws.project_dir
            if len(path) > 28:
                path = "..." + path[-25:]

            cost = fmt_cost(ws.stats.total_cost, currency=ws.currency)
            dates = ""
            if ws.first_active and ws.last_active:
                if ws.first_active == ws.last_active:
                    dates = ws.first_active.strftime("%b %-d")
                else:
                    dates = f"{ws.first_active.strftime('%b %-d')} - {ws.last_active.strftime('%b %-d')}"

            cells = [
                path,
                str(ws.stats.session_count),
                cost,
                f"+{fmt_number(ws.stats.lines_added)}",
                f"-{fmt_number(ws.stats.lines_removed)}",
                fmt_duration(ws.stats.duration_ms),
                dates,
            ]
            for i, cell_text in enumerate(cells):
                pdf.cell(col_widths[i], 6, cell_text, border=1)
            pdf.ln()

    def _hour_histograms(
        self, hist: HourHistogram, sub: Subscription
    ) -> None:
        pdf = self.pdf

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(
            0, 8,
            f"Activity by Hour ({self.tz})",
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.ln(2)

        max_bar_width = 80  # mm

        # Lines changed histogram
        max_lines = max(hist.lines_changed)
        if max_lines > 0:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, "Lines changed", new_x="LMARGIN", new_y="NEXT")
            self._draw_histogram(
                hist.lines_changed,
                max_val=max_lines,
                max_bar_width=max_bar_width,
                color=COLOR_BLUE,
                fmt_value=lambda v: fmt_number(int(v)),
            )
            pdf.ln(3)

        # Cost histogram
        max_cost = max(hist.cost)
        if max_cost > 0:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, "Cost", new_x="LMARGIN", new_y="NEXT")
            self._draw_histogram(
                hist.cost,
                max_val=max_cost,
                max_bar_width=max_bar_width,
                color=COLOR_GOLD,
                fmt_value=lambda v: fmt_cost(v, sub),
            )

    def _draw_histogram(
        self,
        values: list[int] | list[float],
        max_val: float,
        max_bar_width: float,
        color: tuple[int, int, int],
        fmt_value: object,
    ) -> None:
        pdf = self.pdf
        bar_height = 4
        label_width = 15
        value_margin = 2

        pdf.set_font("Helvetica", "", 8)
        for hour, val in enumerate(values):
            if val == 0:
                continue
            bar_width = float(val) / max_val * max_bar_width
            if bar_width < 0.5:
                bar_width = 0.5

            x_start = pdf.get_x() + label_width
            y = pdf.get_y()

            # Hour label
            pdf.set_text_color(*COLOR_DARK)
            pdf.cell(label_width, bar_height + 1, f"{hour:02d}:00")

            # Bar
            pdf.set_fill_color(*color)
            pdf.rect(x_start, y, bar_width, bar_height, style="F")

            # Value label
            pdf.set_xy(x_start + bar_width + value_margin, y)
            pdf.set_text_color(*COLOR_GRAY)
            pdf.cell(30, bar_height + 1, fmt_value(val))  # type: ignore[operator]
            pdf.set_text_color(*COLOR_BLACK)
            pdf.ln(bar_height + 1.5)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    dev_mode = "--dev" in sys.argv

    if dev_mode:
        data_dir = os.path.join(SCRIPT_DIR, "data", "dev")
        csv_path = os.path.join(data_dir, "stats.csv")
        settings_path = os.path.join(data_dir, "settings.json")
    else:
        csv_path = CSV_PATH
        settings_path = SETTINGS_PATH

    # Parse --output
    output_path = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            break
        if arg.startswith("--output="):
            output_path = arg.split("=", 1)[1]
            break

    if output_path is None:
        if dev_mode:
            output_path = os.path.join(SCRIPT_DIR, "data", "dev", "usage-report.pdf")
        else:
            output_path = os.path.join(DATA_DIR, "usage-report.pdf")

    compact_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "compact.py")]
    if dev_mode:
        compact_cmd.append("--dev")
    subprocess.run(compact_cmd)

    settings = load_settings(settings_path)
    today = datetime.now(settings.tz).date()

    finished = find_finished_subscriptions(settings.subscriptions, today)
    if not finished:
        print("No finished subscriptions found.", file=sys.stderr)
        return 1

    df = load_sessions(csv_path)
    if df.empty:
        print("No session data found.", file=sys.stderr)
        return 1

    reports = [build_subscription_report(df, sub, settings.tz) for sub in finished]

    pdf = PdfReport(settings.tz)
    pdf.generate(reports, settings)
    pdf.save(output_path)
    print(f"Report saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
