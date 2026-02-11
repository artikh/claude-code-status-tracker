"""Tests for report_pdf.py."""

from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

import pytest

from conftest import make_sessions_df
from report_pdf import (
    PdfReport,
    SubscriptionReport,
    build_subscription_report,
    find_finished_subscriptions,
)
from usage import (
    HourHistogram,
    PeriodStats,
    Settings,
    Subscription,
    WorkspaceStats,
)

UTC = ZoneInfo("UTC")


# ── find_finished_subscriptions ─────────────────────────────────────────────


class TestFindFinishedSubscriptions:
    def test_all_finished(self) -> None:
        subs = [
            Subscription("A", 100, "USD", date(2025, 1, 1), date(2025, 1, 31)),
            Subscription("B", 100, "USD", date(2025, 2, 1), date(2025, 2, 28)),
        ]
        result = find_finished_subscriptions(subs, date(2025, 3, 1))
        assert len(result) == 2
        assert result[0].plan == "A"
        assert result[1].plan == "B"

    def test_none_finished(self) -> None:
        subs = [
            Subscription("A", 100, "USD", date(2026, 3, 1), date(2026, 3, 31)),
        ]
        result = find_finished_subscriptions(subs, date(2026, 3, 15))
        assert result == []

    def test_mixed(self) -> None:
        subs = [
            Subscription("Old", 100, "USD", date(2025, 1, 1), date(2025, 1, 31)),
            Subscription("Current", 100, "USD", date(2026, 2, 1), date(2026, 2, 28)),
        ]
        result = find_finished_subscriptions(subs, date(2026, 2, 15))
        assert len(result) == 1
        assert result[0].plan == "Old"

    def test_boundary_end_equals_today_not_finished(self) -> None:
        subs = [
            Subscription("Edge", 100, "USD", date(2026, 2, 1), date(2026, 2, 11)),
        ]
        result = find_finished_subscriptions(subs, date(2026, 2, 11))
        assert result == []

    def test_boundary_end_before_today_finished(self) -> None:
        subs = [
            Subscription("Done", 100, "USD", date(2026, 2, 1), date(2026, 2, 10)),
        ]
        result = find_finished_subscriptions(subs, date(2026, 2, 11))
        assert len(result) == 1

    def test_empty_list(self) -> None:
        assert find_finished_subscriptions([], date(2026, 2, 11)) == []

    def test_sort_order(self) -> None:
        subs = [
            Subscription("Later", 100, "USD", date(2025, 6, 1), date(2025, 6, 30)),
            Subscription("Earlier", 100, "USD", date(2025, 1, 1), date(2025, 1, 31)),
            Subscription("Middle", 100, "USD", date(2025, 3, 1), date(2025, 3, 31)),
        ]
        result = find_finished_subscriptions(subs, date(2026, 1, 1))
        assert [r.plan for r in result] == ["Earlier", "Middle", "Later"]


# ── build_subscription_report ───────────────────────────────────────────────


class TestBuildSubscriptionReport:
    def test_filters_to_date_range(self) -> None:
        sub = Subscription("A", 100, "USD", date(2026, 2, 1), date(2026, 2, 28))
        df = make_sessions_df([
            {"first_active": "2026-02-05T10:00:00+00:00", "cost_usd": 5.0},
            {"first_active": "2026-02-15T10:00:00+00:00", "cost_usd": 3.0},
            {"first_active": "2026-01-15T10:00:00+00:00", "cost_usd": 99.0},  # outside
        ])
        report = build_subscription_report(df, sub, UTC)
        assert report.stats.session_count == 2
        assert report.stats.total_cost == 8.0
        assert report.subscription == sub

    def test_empty_period(self) -> None:
        sub = Subscription("A", 100, "USD", date(2025, 6, 1), date(2025, 6, 30))
        df = make_sessions_df([
            {"first_active": "2026-02-11T10:00:00+00:00", "cost_usd": 5.0},
        ])
        report = build_subscription_report(df, sub, UTC)
        assert report.stats.session_count == 0
        assert report.stats.total_cost == 0.0

    def test_workspace_stats_scoped(self) -> None:
        sub = Subscription("A", 100, "GBP", date(2026, 2, 1), date(2026, 2, 28), 1.25)
        df = make_sessions_df([
            {"first_active": "2026-02-05T10:00:00+00:00", "cost_usd": 12.50,
             "project_dir": "/src/proj"},
            {"first_active": "2026-02-10T10:00:00+00:00", "cost_usd": 6.25,
             "project_dir": "/src/other"},
        ])
        report = build_subscription_report(df, sub, UTC)
        assert len(report.workspaces) == 2
        # All workspaces should use GBP
        for ws in report.workspaces:
            assert ws.currency == "GBP"

    def test_hour_histogram_present(self) -> None:
        sub = Subscription("A", 100, "USD", date(2026, 2, 1), date(2026, 2, 28))
        df = make_sessions_df([
            {"first_active": "2026-02-05T14:00:00+00:00", "cost_usd": 5.0,
             "lines_added": 100, "lines_removed": 20},
        ])
        report = build_subscription_report(df, sub, UTC)
        assert report.hour_histogram.has_data
        assert report.hour_histogram.lines_changed[14] == 120


# ── PDF generation integration ──────────────────────────────────────────────


class TestPdfGeneration:
    def _make_report(
        self, plan: str = "Max 20x", sessions: int = 3
    ) -> SubscriptionReport:
        sub = Subscription(plan, 150, "GBP", date(2026, 1, 1), date(2026, 1, 31), 1.25)
        return SubscriptionReport(
            subscription=sub,
            stats=PeriodStats(
                session_count=sessions,
                total_cost=50.0,
                lines_added=1000,
                lines_removed=200,
                duration_ms=3600000,
                api_duration_ms=1800000,
                total_input_tokens=100000,
                total_output_tokens=50000,
                avg_context_used_pct=30.0,
                extended_context_count=0,
            ),
            workspaces=[
                WorkspaceStats(
                    project_dir="/src/my-project",
                    stats=PeriodStats(
                        session_count=sessions,
                        total_cost=40.0,
                        lines_added=800,
                        lines_removed=150,
                        duration_ms=3000000,
                        api_duration_ms=1500000,
                        total_input_tokens=80000,
                        total_output_tokens=40000,
                    ),
                    currency="GBP",
                    first_active=date(2026, 1, 5),
                    last_active=date(2026, 1, 25),
                ),
            ],
            hour_histogram=HourHistogram(
                lines_changed=[0] * 9 + [300, 500, 200] + [0] * 12,
                cost=[0] * 9 + [3.0, 5.0, 2.0] + [0] * 12,
            ),
        )

    def test_generates_valid_pdf(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        report = self._make_report()
        settings = Settings(tz=UTC, subscriptions=[report.subscription])
        pdf = PdfReport(UTC)
        pdf.generate([report], settings)
        path = str(tmp_path / "test.pdf")
        pdf.save(path)

        with open(path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

    def test_empty_subscription_no_crash(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        sub = Subscription("Empty", 100, "USD", date(2025, 6, 1), date(2025, 6, 30))
        report = SubscriptionReport(
            subscription=sub,
            stats=PeriodStats(),
            workspaces=[],
            hour_histogram=HourHistogram(),
        )
        settings = Settings(tz=UTC, subscriptions=[sub])
        pdf = PdfReport(UTC)
        pdf.generate([report], settings)
        path = str(tmp_path / "empty.pdf")
        pdf.save(path)

        with open(path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

    def test_multiple_subscriptions_multiple_pages(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        reports = [self._make_report(plan=f"Plan {i}") for i in range(3)]
        settings = Settings(
            tz=UTC,
            subscriptions=[r.subscription for r in reports],
        )
        pdf = PdfReport(UTC)
        pdf.generate(reports, settings)
        path = str(tmp_path / "multi.pdf")
        pdf.save(path)

        # Title page + 3 subscription pages = at least 4 pages
        assert pdf.pdf.pages_count >= 4
