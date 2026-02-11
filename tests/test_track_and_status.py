"""Tests for track-and-status.py."""

import importlib
import io
import json
import sys
from pathlib import Path

import pytest

# The module has a hyphenated filename, so import via importlib
SCRIPT = Path(__file__).resolve().parent.parent / "track-and-status.py"
spec = importlib.util.spec_from_file_location("track_and_status", SCRIPT)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

build_bar = mod.build_bar
get_stage = mod.get_stage
log_event = mod.log_event
main = mod.main

# ── build_bar ──────────────────────────────────────────────────────────


class TestBuildBar:
    def test_zero_percent(self):
        bar, color = build_bar(0)
        assert bar == "░" * 10
        assert color == mod.GREEN

    def test_fifty_percent(self):
        bar, color = build_bar(50)
        assert bar.count("█") == 5
        assert bar.count("░") == 5
        assert color == mod.GREEN

    def test_seventy_five_percent_yellow(self):
        bar, color = build_bar(75)
        assert color == mod.YELLOW

    def test_ninety_five_percent_red(self):
        bar, color = build_bar(95)
        assert color == mod.RED

    def test_custom_width(self):
        bar, _ = build_bar(50, width=20)
        assert len(bar) == 20
        assert bar.count("█") == 10

    def test_hundred_percent(self):
        bar, color = build_bar(100)
        assert bar == "█" * 10
        assert color == mod.RED


# ── get_stage ──────────────────────────────────────────────────────────


class TestGetStage:
    def test_valid_stage(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "CACHE_FILE", str(tmp_path / "stage-cache"))
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text(
            "# Project\n\n## Current Stage\n\nStage 1: Scaffolding\n"
        )
        assert get_stage(str(tmp_path)) == "Stage 1: Scaffolding"

    def test_no_claude_md(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "CACHE_FILE", str(tmp_path / "stage-cache"))
        assert get_stage(str(tmp_path)) == ""

    def test_no_stage_section(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "CACHE_FILE", str(tmp_path / "stage-cache"))
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# Project\n\nSome other content.\n")
        assert get_stage(str(tmp_path)) == ""

    def test_cache_is_used(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "stage-cache"
        monkeypatch.setattr(mod, "CACHE_FILE", str(cache_file))
        # Prime the cache
        cache_file.write_text("Stage 2: Cached")
        # Make it fresh (mtime is now)
        result = get_stage(str(tmp_path))
        assert result == "Stage 2: Cached"


# ── log_event ──────────────────────────────────────────────────────────


class TestLogEvent:
    def test_creates_dir_and_writes_jsonl(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "logs"
        log_file = log_dir / "session.jsonl"
        monkeypatch.setattr(mod, "LOG_DIR", str(log_dir))
        monkeypatch.setattr(mod, "LOG_FILE", str(log_file))

        result = log_event({"key": "val"}, "sess-1")
        assert result is None
        assert log_dir.exists()

        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["session_id"] == "sess-1"
        assert entry["data"] == {"key": "val"}
        assert "timestamp" in entry

    def test_append_behavior(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "logs"
        log_file = log_dir / "session.jsonl"
        monkeypatch.setattr(mod, "LOG_DIR", str(log_dir))
        monkeypatch.setattr(mod, "LOG_FILE", str(log_file))

        log_event({"n": 1}, "s1")
        log_event({"n": 2}, "s2")

        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_failure_returns_error_string(self, tmp_path, monkeypatch):
        # Point LOG_DIR to a file (not a directory) so makedirs fails
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file")
        monkeypatch.setattr(mod, "LOG_DIR", str(blocker / "subdir"))
        monkeypatch.setattr(mod, "LOG_FILE", str(blocker / "subdir" / "log.jsonl"))

        result = log_event({}, "x")
        assert isinstance(result, str)
        assert len(result) > 0


# ── main (integration) ────────────────────────────────────────────────


class TestMain:
    def _make_input(self, **overrides):
        data = {
            "model": {"display_name": "Opus"},
            "context_window": {"used_percentage": 17},
            "cost": {
                "total_cost_usd": 0.45,
                "total_lines_added": 156,
                "total_lines_removed": 23,
            },
            "workspace": {"current_dir": "/tmp/fake"},
            "session_id": "test-sess",
        }
        data.update(overrides)
        return json.dumps(data)

    def test_output_contains_expected_parts(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(mod, "LOG_DIR", str(log_dir))
        monkeypatch.setattr(mod, "LOG_FILE", str(log_dir / "session.jsonl"))
        monkeypatch.setattr(mod, "CACHE_FILE", str(tmp_path / "stage-cache"))
        monkeypatch.setattr("sys.stdin", io.StringIO(self._make_input()))

        captured = io.StringIO()
        monkeypatch.setattr("sys.stdout", captured)
        main()

        output = captured.getvalue()
        assert "Opus" in output
        assert "17%" in output
        assert "$0.45" in output
        assert "+156" in output
        assert "-23" in output
        assert "[LOG ERR]" not in output

    def test_no_log_error_on_success(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(mod, "LOG_DIR", str(log_dir))
        monkeypatch.setattr(mod, "LOG_FILE", str(log_dir / "session.jsonl"))
        monkeypatch.setattr(mod, "CACHE_FILE", str(tmp_path / "stage-cache"))
        monkeypatch.setattr("sys.stdin", io.StringIO(self._make_input()))

        captured = io.StringIO()
        monkeypatch.setattr("sys.stdout", captured)
        main()

        assert "[LOG ERR]" not in captured.getvalue()
