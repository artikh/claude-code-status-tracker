"""Tests for usage.py."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import usage
from usage import (
    HourHistogram,
    PeriodStats,
    Settings,
    Subscription,
    UsageReport,
    build_report,
    compute_hour_histogram,
    compute_stats,
    compute_time_periods,
    compute_workspace_stats,
    filter_by_date_range,
    find_active_subscription,
    load_sessions,
    load_settings,
    render_terminal,
)


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


UTC = ZoneInfo("UTC")


class TestComputeStats:
    def test_empty_dataframe(self) -> None:
        df = make_sessions_df([])
        stats = compute_stats(df)
        assert stats.session_count == 0
        assert stats.total_cost == 0.0
        assert stats.cost_per_line is None
        assert stats.avg_cost_per_session is None

    def test_single_session(self) -> None:
        df = make_sessions_df([{"cost_usd": 5.0, "lines_added": 200, "lines_removed": 50}])
        stats = compute_stats(df)
        assert stats.session_count == 1
        assert stats.total_cost == 5.0
        assert stats.lines_added == 200
        assert stats.lines_removed == 50
        assert stats.cost_per_line == pytest.approx(5.0 / 250)
        assert stats.avg_cost_per_session == 5.0

    def test_multi_session(self) -> None:
        df = make_sessions_df([
            {"cost_usd": 3.0, "lines_added": 100, "lines_removed": 10},
            {"cost_usd": 2.0, "lines_added": 50, "lines_removed": 5},
        ])
        stats = compute_stats(df)
        assert stats.session_count == 2
        assert stats.total_cost == 5.0
        assert stats.lines_added == 150
        assert stats.lines_removed == 15
        assert stats.avg_cost_per_session == 2.5

    def test_zero_lines_edge_case(self) -> None:
        df = make_sessions_df([{"cost_usd": 1.0, "lines_added": 0, "lines_removed": 0}])
        stats = compute_stats(df)
        assert stats.cost_per_line is None

    def test_api_wait_ratio(self) -> None:
        df = make_sessions_df([{"duration_ms": 10000, "api_duration_ms": 4000}])
        stats = compute_stats(df)
        assert stats.api_wait_ratio == pytest.approx(0.4)

    def test_output_tokens_per_dollar(self) -> None:
        df = make_sessions_df([{"cost_usd": 2.0, "total_output_tokens": 10000}])
        stats = compute_stats(df)
        assert stats.output_tokens_per_dollar == pytest.approx(5000.0)

    def test_zero_cost_tokens_per_dollar(self) -> None:
        df = make_sessions_df([{"cost_usd": 0.0}])
        stats = compute_stats(df)
        assert stats.output_tokens_per_dollar is None

    def test_extended_context_count(self) -> None:
        df = make_sessions_df([
            {"exceeds_200k_tokens": True},
            {"exceeds_200k_tokens": False},
            {"exceeds_200k_tokens": True},
        ])
        stats = compute_stats(df)
        assert stats.extended_context_count == 2


class TestTimePeriods:
    def test_today_filter(self) -> None:
        tz = ZoneInfo("UTC")
        today = date.today()
        today_ts = datetime.combine(today, datetime.min.time(), tzinfo=tz).isoformat()
        yesterday_ts = datetime.combine(
            today - timedelta(days=1), datetime.min.time(), tzinfo=tz
        ).isoformat()

        df = make_sessions_df([
            {"first_active": today_ts, "cost_usd": 5.0},
            {"first_active": yesterday_ts, "cost_usd": 3.0},
        ])
        periods = compute_time_periods(df, tz, None)

        # First period is Today
        today_period = periods[0]
        assert "Today" in today_period.name
        assert today_period.stats.session_count == 1
        assert today_period.stats.total_cost == 5.0

    def test_billing_period_from_subscription(self) -> None:
        tz = ZoneInfo("UTC")
        df = make_sessions_df([
            {"first_active": "2026-02-01T10:00:00+00:00", "cost_usd": 5.0},
            {"first_active": "2026-02-15T10:00:00+00:00", "cost_usd": 3.0},
            {"first_active": "2026-01-01T10:00:00+00:00", "cost_usd": 99.0},
        ])
        sub = Subscription(
            plan="Max 20x", cost=150.0, currency="GBP",
            start=date(2026, 1, 29), end=date(2026, 2, 28), usd_rate=1.24,
        )
        periods = compute_time_periods(df, tz, sub)

        # Last period should be billing
        billing = periods[-1]
        assert "Billing Period" in billing.name
        assert billing.stats.session_count == 2
        assert billing.stats.total_cost == 8.0

    def test_no_subscription(self) -> None:
        tz = ZoneInfo("UTC")
        df = make_sessions_df([{"first_active": "2026-02-11T10:00:00+00:00"}])
        periods = compute_time_periods(df, tz, None)
        # Should have 4 periods: Today, Yesterday, This Week, This Month
        assert len(periods) == 4
        assert all("Billing" not in p.name for p in periods)


class TestWorkspaceStats:
    def test_grouping_and_sort_no_subs(self) -> None:
        df = make_sessions_df([
            {"project_dir": "/src/cheap", "cost_usd": 1.0},
            {"project_dir": "/src/expensive", "cost_usd": 10.0},
            {"project_dir": "/src/expensive", "cost_usd": 5.0},
        ])
        workspaces = compute_workspace_stats(df, [])
        assert len(workspaces) == 2
        assert workspaces[0].project_dir == "/src/expensive"
        assert workspaces[0].stats.total_cost == 15.0
        assert workspaces[0].stats.session_count == 2
        assert workspaces[0].currency == "USD"
        assert workspaces[1].project_dir == "/src/cheap"

    def test_converts_with_subscription_rate(self) -> None:
        sub = Subscription("Max", 150, "GBP", date(2026, 2, 1), date(2026, 2, 28), 1.25)
        df = make_sessions_df([
            {"project_dir": "/src/proj", "cost_usd": 12.50,
             "first_active": "2026-02-11T10:00:00+00:00"},
        ])
        workspaces = compute_workspace_stats(df, [sub])
        assert len(workspaces) == 1
        # $12.50 / 1.25 = £10.00
        assert workspaces[0].stats.total_cost == pytest.approx(10.0)
        assert workspaces[0].currency == "GBP"

    def test_multi_subscription_conversion(self) -> None:
        subs = [
            Subscription("A", 100, "GBP", date(2026, 1, 1), date(2026, 1, 31), 1.20),
            Subscription("B", 150, "GBP", date(2026, 2, 1), date(2026, 2, 28), 1.25),
        ]
        df = make_sessions_df([
            {"project_dir": "/src/proj", "cost_usd": 12.00,
             "first_active": "2026-01-15T10:00:00+00:00"},  # sub A: /1.20 = £10.00
            {"project_dir": "/src/proj", "cost_usd": 12.50,
             "first_active": "2026-02-11T10:00:00+00:00"},  # sub B: /1.25 = £10.00
        ])
        workspaces = compute_workspace_stats(df, subs)
        assert len(workspaces) == 1
        assert workspaces[0].stats.total_cost == pytest.approx(20.0)
        assert workspaces[0].currency == "GBP"
        assert workspaces[0].first_active == date(2026, 1, 15)
        assert workspaces[0].last_active == date(2026, 2, 11)

    def test_date_range_single_day(self) -> None:
        df = make_sessions_df([
            {"project_dir": "/src/proj",
             "first_active": "2026-02-11T09:00:00+00:00"},
            {"project_dir": "/src/proj",
             "first_active": "2026-02-11T14:00:00+00:00"},
        ])
        workspaces = compute_workspace_stats(df, [])
        assert workspaces[0].first_active == date(2026, 2, 11)
        assert workspaces[0].last_active == date(2026, 2, 11)

    def test_empty_project_dir_skipped(self) -> None:
        df = make_sessions_df([
            {"project_dir": "", "cost_usd": 1.0},
            {"project_dir": "/src/real", "cost_usd": 2.0},
        ])
        workspaces = compute_workspace_stats(df, [])
        assert len(workspaces) == 1
        assert workspaces[0].project_dir == "/src/real"

    def test_empty_dataframe(self) -> None:
        df = make_sessions_df([])
        workspaces = compute_workspace_stats(df, [])
        assert workspaces == []


class TestHourHistogram:
    def test_lines_changed_by_hour(self) -> None:
        tz = ZoneInfo("UTC")
        df = make_sessions_df([
            {"first_active": "2026-02-11T09:00:00+00:00",
             "lines_added": 50, "lines_removed": 10},
            {"first_active": "2026-02-11T09:30:00+00:00",
             "lines_added": 30, "lines_removed": 5},
            {"first_active": "2026-02-11T14:00:00+00:00",
             "lines_added": 100, "lines_removed": 20},
        ])
        hist = compute_hour_histogram(df, tz)
        assert hist.lines_changed[9] == 95  # 50+10+30+5
        assert hist.lines_changed[14] == 120  # 100+20
        assert sum(hist.lines_changed) == 215

    def test_cost_by_hour(self) -> None:
        tz = ZoneInfo("UTC")
        df = make_sessions_df([
            {"first_active": "2026-02-11T09:00:00+00:00", "cost_usd": 3.0},
            {"first_active": "2026-02-11T09:30:00+00:00", "cost_usd": 2.0},
            {"first_active": "2026-02-11T14:00:00+00:00", "cost_usd": 5.0},
        ])
        hist = compute_hour_histogram(df, tz)
        assert hist.cost[9] == pytest.approx(5.0)
        assert hist.cost[14] == pytest.approx(5.0)

    def test_timezone_shift(self) -> None:
        tz = ZoneInfo("US/Pacific")
        df = make_sessions_df([
            {"first_active": "2026-02-11T18:00:00+00:00",
             "lines_added": 100, "lines_removed": 20},
        ])
        hist = compute_hour_histogram(df, tz)
        assert hist.lines_changed[10] == 120  # 10:00 PST

    def test_has_data(self) -> None:
        tz = ZoneInfo("UTC")
        df = make_sessions_df([
            {"first_active": "2026-02-11T10:00:00+00:00",
             "lines_added": 100, "lines_removed": 10, "cost_usd": 5.0},
        ])
        hist = compute_hour_histogram(df, tz)
        assert hist.has_data is True

    def test_empty_data(self) -> None:
        tz = ZoneInfo("UTC")
        df = make_sessions_df([])
        hist = compute_hour_histogram(df, tz)
        assert hist.has_data is False
        assert sum(hist.lines_changed) == 0
        assert sum(hist.cost) == 0


class TestLoadSettings:
    def test_valid_settings(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({
            "subscriptions": [
                {"plan": "Max 20x", "cost": 150, "currency": "GBP",
                 "usd_rate": 1.24,
                 "start": "2026-01-29", "end": "2026-02-28"},
            ],
            "timezone": "US/Pacific",
        }))
        s = load_settings(str(path))
        assert len(s.subscriptions) == 1
        sub = s.subscriptions[0]
        assert sub.plan == "Max 20x"
        assert sub.cost == 150.0
        assert sub.usd_rate == 1.24
        assert sub.cost_usd == pytest.approx(186.0)
        assert sub.start == date(2026, 1, 29)
        assert sub.end == date(2026, 2, 28)
        assert str(s.tz) == "US/Pacific"

    def test_usd_subscription_no_rate_needed(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({
            "subscriptions": [
                {"plan": "Pro", "cost": 200, "currency": "USD",
                 "start": "2026-01-01", "end": "2026-01-31"},
            ],
        }))
        s = load_settings(str(path))
        assert len(s.subscriptions) == 1
        assert s.subscriptions[0].usd_rate == 1.0
        assert s.subscriptions[0].cost_usd == 200.0

    def test_non_usd_without_rate_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({
            "subscriptions": [
                {"plan": "Bad", "cost": 100, "currency": "GBP",
                 "start": "2026-01-01", "end": "2026-01-31"},
            ],
        }))
        s = load_settings(str(path))
        assert s.subscriptions == []

    def test_multiple_subscriptions(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({
            "subscriptions": [
                {"plan": "Plan A", "cost": 100, "currency": "GBP",
                 "usd_rate": 1.24,
                 "start": "2026-01-01", "end": "2026-01-31"},
                {"plan": "Plan B", "cost": 150, "currency": "GBP",
                 "usd_rate": 1.25,
                 "start": "2026-02-01", "end": "2026-02-28"},
            ],
        }))
        s = load_settings(str(path))
        assert len(s.subscriptions) == 2

    def test_missing_file(self, tmp_path: Path) -> None:
        s = load_settings(str(tmp_path / "nonexistent.json"))
        assert s.subscriptions == []
        assert str(s.tz) == "UTC"

    def test_missing_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text("{}")
        s = load_settings(str(path))
        assert s.subscriptions == []

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text("NOT JSON")
        s = load_settings(str(path))
        assert s.subscriptions == []

    def test_invalid_subscription_date(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({
            "subscriptions": [
                {"plan": "Bad", "cost": 100, "currency": "GBP",
                 "usd_rate": 1.24,
                 "start": "not-a-date", "end": "2026-02-28"},
            ],
        }))
        s = load_settings(str(path))
        # Entire subscriptions list rejected on parse error
        assert s.subscriptions == []

    def test_overlapping_subscriptions(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({
            "subscriptions": [
                {"plan": "A", "cost": 100, "currency": "GBP",
                 "usd_rate": 1.24,
                 "start": "2026-01-01", "end": "2026-02-15"},
                {"plan": "B", "cost": 150, "currency": "GBP",
                 "usd_rate": 1.24,
                 "start": "2026-02-01", "end": "2026-02-28"},
            ],
        }))
        s = load_settings(str(path))
        # Overlapping subs are rejected
        assert s.subscriptions == []

    def test_invalid_timezone(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"timezone": "Not/A/Timezone"}))
        s = load_settings(str(path))
        assert str(s.tz) == "UTC"


class TestLoadSessions:
    def test_loads_csv(self, tmp_path: Path) -> None:
        csv = tmp_path / "stats.csv"
        csv.write_text(
            "session_id,first_active,last_active,cost_usd,lines_added,lines_removed,"
            "duration_ms,api_duration_ms,total_input_tokens,total_output_tokens,"
            "context_window_size,context_used_pct,exceeds_200k_tokens\n"
            "s1,2026-02-11T10:00:00+00:00,2026-02-11T10:30:00+00:00,5.0,100,20,"
            "600000,300000,10000,5000,200000,25,False\n"
        )
        df = load_sessions(str(csv))
        assert len(df) == 1
        assert df.iloc[0]["cost_usd"] == 5.0
        assert df.iloc[0]["exceeds_200k_tokens"] == False  # noqa: E712

    def test_missing_csv(self, tmp_path: Path) -> None:
        df = load_sessions(str(tmp_path / "nope.csv"))
        assert df.empty

    def test_empty_csv(self, tmp_path: Path) -> None:
        csv = tmp_path / "stats.csv"
        csv.write_text(
            "session_id,first_active,last_active,cost_usd\n"
        )
        df = load_sessions(str(csv))
        assert df.empty


class TestSubscription:
    def test_find_active_subscription(self) -> None:
        subs = [
            Subscription("A", 100, "GBP", date(2026, 1, 1), date(2026, 1, 31), 1.24),
            Subscription("B", 150, "GBP", date(2026, 2, 1), date(2026, 2, 28), 1.24),
        ]
        assert find_active_subscription(subs, date(2026, 1, 15)) == subs[0]
        assert find_active_subscription(subs, date(2026, 2, 11)) == subs[1]

    def test_no_active_subscription(self) -> None:
        subs = [
            Subscription("A", 100, "GBP", date(2026, 1, 1), date(2026, 1, 31), 1.24),
        ]
        assert find_active_subscription(subs, date(2026, 3, 1)) is None

    def test_boundary_dates(self) -> None:
        sub = Subscription("A", 100, "GBP", date(2026, 1, 29), date(2026, 2, 28), 1.24)
        assert find_active_subscription([sub], date(2026, 1, 29)) == sub
        assert find_active_subscription([sub], date(2026, 2, 28)) == sub
        assert find_active_subscription([sub], date(2026, 1, 28)) is None

    def test_cost_usd_conversion(self) -> None:
        sub = Subscription("Max", 150, "GBP", date(2026, 1, 1), date(2026, 1, 31), 1.24)
        assert sub.cost_usd == pytest.approx(186.0)

    def test_cost_usd_defaults_to_1(self) -> None:
        sub = Subscription("Pro", 200, "USD", date(2026, 1, 1), date(2026, 1, 31))
        assert sub.cost_usd == 200.0

    def test_overlap_detection(self) -> None:
        subs = [
            Subscription("A", 100, "GBP", date(2026, 1, 1), date(2026, 2, 15), 1.24),
            Subscription("B", 150, "GBP", date(2026, 2, 1), date(2026, 2, 28), 1.24),
        ]
        with pytest.raises(ValueError, match="Overlapping"):
            usage._check_overlaps(subs)

    def test_no_overlap_adjacent(self) -> None:
        subs = [
            Subscription("A", 100, "GBP", date(2026, 1, 1), date(2026, 1, 31), 1.24),
            Subscription("B", 150, "GBP", date(2026, 2, 1), date(2026, 2, 28), 1.24),
        ]
        usage._check_overlaps(subs)

    def test_empty_list(self) -> None:
        assert find_active_subscription([], date(2026, 2, 11)) is None
        usage._check_overlaps([])


class TestRenderTerminal:
    _sub = Subscription(
        plan="Max 20x", cost=150.0, currency="GBP",
        start=date(2026, 1, 29), end=date(2026, 2, 28), usd_rate=1.24,
    )

    def _make_report(self, sub: Subscription | None = _sub) -> UsageReport:
        """Create a minimal report for rendering tests."""
        tz = ZoneInfo("UTC")
        stats = PeriodStats(
            session_count=5,
            total_cost=13.71,
            lines_added=2429,
            lines_removed=506,
            duration_ms=6540000,
            api_duration_ms=3060000,
            total_input_tokens=347000,
            total_output_tokens=155000,
            avg_context_used_pct=24.0,
            extended_context_count=0,
        )
        return UsageReport(
            generated_at=datetime(2026, 2, 11, 10, 30, tzinfo=tz),
            tz=tz,
            time_periods=[
                usage.TimePeriod(
                    name="Today (Feb 11)",
                    start=date(2026, 2, 11),
                    end=date(2026, 2, 11),
                    stats=stats,
                ),
                usage.TimePeriod(
                    name="Yesterday (Feb 10)",
                    start=date(2026, 2, 10),
                    end=date(2026, 2, 10),
                    stats=PeriodStats(),
                ),
            ],
            workspaces=[
                usage.WorkspaceStats(
                    project_dir="/src/project",
                    stats=stats,
                ),
            ],
            hour_histogram=HourHistogram(
                lines_changed=[0] * 9 + [300, 500] + [0] * 13,
                cost=[0] * 9 + [3.0, 5.0] + [0] * 13,
            ),
            subscription=sub,
        )

    def test_contains_expected_text(self) -> None:
        report = self._make_report()
        output = render_terminal(report, color=True)
        assert "Claude Code Usage Report" in output
        assert "Today (Feb 11)" in output
        # $13.71 / 1.24 = £11.06
        assert "£11.06" in output
        assert "+2,429" in output
        assert "-506" in output
        assert "Sessions: 5" in output
        assert "No sessions" in output  # Yesterday
        assert "/src/project" in output
        assert "Lines changed" in output
        assert "Cost" in output

    def test_subscription_header(self) -> None:
        report = self._make_report()
        output = render_terminal(report, color=False)
        assert "Max 20x" in output
        assert "£150.00" in output

    def test_subscription_percentage(self) -> None:
        report = self._make_report()
        output = render_terminal(report, color=False)
        # $13.71 / (£150 * 1.24 = $186) = 7%
        assert "7% Max 20x" in output

    def test_over_budget_warning(self) -> None:
        over_sub = Subscription(
            plan="Cheap", cost=10.0, currency="USD",
            start=date(2026, 1, 29), end=date(2026, 2, 28),
        )
        report = self._make_report(sub=over_sub)
        output = render_terminal(report, color=False)
        assert "OVER BUDGET" in output

    def test_no_subscription(self) -> None:
        report = self._make_report(sub=None)
        output = render_terminal(report, color=False)
        assert "£" not in output
        assert "OVER BUDGET" not in output

    def test_no_color_strips_ansi(self) -> None:
        report = self._make_report()
        output = render_terminal(report, color=False)
        assert "\033[" not in output
        assert "Claude Code Usage Report" in output
        assert "£11.06" in output


class TestMainCli:
    def test_no_active_subscription_exits_with_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Settings with an expired subscription
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps({
            "subscriptions": [
                {"plan": "Old Plan", "cost": 100, "currency": "USD",
                 "start": "2020-01-01", "end": "2020-01-31"},
            ],
        }))
        csv_path = tmp_path / "stats.csv"
        csv_path.write_text("session_id,cost_usd\ns1,1.0\n")

        monkeypatch.setattr(usage, "CSV_PATH", str(csv_path))
        monkeypatch.setattr(usage, "SETTINGS_PATH", str(settings_path))
        # Ensure --dev is not in argv
        monkeypatch.setattr("sys.argv", ["usage.py"])

        rc = usage.main()
        assert rc == 1

        captured = capsys.readouterr()
        assert "no active subscription" in captured.err
        assert "Old Plan" in captured.err
        assert "2020-01-31" in captured.err

    def test_no_subscriptions_at_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}")
        csv_path = tmp_path / "stats.csv"
        csv_path.write_text("session_id,cost_usd\ns1,1.0\n")

        monkeypatch.setattr(usage, "CSV_PATH", str(csv_path))
        monkeypatch.setattr(usage, "SETTINGS_PATH", str(settings_path))
        monkeypatch.setattr("sys.argv", ["usage.py"])

        rc = usage.main()
        assert rc == 1

        captured = capsys.readouterr()
        assert "no active subscription" in captured.err
        assert "no subscriptions configured" in captured.err


class TestFormatHelpers:
    def test_fmt_cost_usd_default(self) -> None:
        assert usage._fmt_cost(0.0) == "$0.00"
        assert usage._fmt_cost(13.71) == "$13.71"
        assert usage._fmt_cost(0.0047) == "$0.0047"

    def test_fmt_cost_converts_to_sub_currency(self) -> None:
        sub = Subscription("Max", 150, "GBP", date(2026, 1, 1), date(2026, 1, 31), 1.25)
        # $12.50 / 1.25 = £10.00
        assert usage._fmt_cost(12.50, sub) == "£10.00"
        # $1.25 / 1.25 = £1.00
        assert usage._fmt_cost(1.25, sub) == "£1.00"

    def test_fmt_duration(self) -> None:
        assert usage._fmt_duration(0) == "0s"
        assert usage._fmt_duration(45000) == "45s"
        assert usage._fmt_duration(120000) == "2m"
        assert usage._fmt_duration(150000) == "2m 30s"
        assert usage._fmt_duration(3600000) == "1h"
        assert usage._fmt_duration(6540000) == "1h 49m"

    def test_fmt_tokens(self) -> None:
        assert usage._fmt_tokens(500) == "500"
        assert usage._fmt_tokens(1000) == "1K"
        assert usage._fmt_tokens(347000) == "347K"
        assert usage._fmt_tokens(1500000) == "1.5M"

    def test_fmt_number(self) -> None:
        assert usage._fmt_number(0) == "0"
        assert usage._fmt_number(2429) == "2,429"
        assert usage._fmt_number(1000000) == "1,000,000"
