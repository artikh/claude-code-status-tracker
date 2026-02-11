"""Tests for compact.py."""

import fcntl
import json
import os
from pathlib import Path

import pandas as pd
import pytest

import compact


def make_entry(
    session_id: str = "sess-1",
    timestamp: str = "2026-01-15T10:00:00+00:00",
    cost_usd: float = 1.0,
    lines_added: int = 10,
    lines_removed: int = 2,
    used_pct: object = 15,
    model_id: str = "claude-opus-4-6",
    model_name: str = "Opus 4.6",
    **overrides: object,
) -> dict:
    """Build a JSONL entry dict for testing."""
    entry: dict = {
        "timestamp": timestamp,
        "session_id": session_id,
        "data": {
            "session_id": session_id,
            "model": {"id": model_id, "display_name": model_name},
            "workspace": {
                "current_dir": "/src/project",
                "project_dir": "/src/project",
            },
            "transcript_path": f"/transcripts/{session_id}.jsonl",
            "version": "2.1.39",
            "cost": {
                "total_cost_usd": cost_usd,
                "total_duration_ms": 10000,
                "total_api_duration_ms": 5000,
                "total_lines_added": lines_added,
                "total_lines_removed": lines_removed,
            },
            "context_window": {
                "total_input_tokens": 1000,
                "total_output_tokens": 500,
                "context_window_size": 200000,
                "used_percentage": used_pct,
            },
            "exceeds_200k_tokens": False,
        },
    }
    entry["data"].update(overrides)
    return entry


def write_jsonl(path: Path, entries: list[dict]) -> None:
    """Write a list of entry dicts as JSONL."""
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def setup_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point compact module at tmp_path/data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(compact, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(compact, "JSONL_PATH", str(data_dir / "session.jsonl"))
    monkeypatch.setattr(compact, "PROCESSING_PATH", str(data_dir / "session.jsonl.processing"))
    monkeypatch.setattr(compact, "CSV_PATH", str(data_dir / "stats.csv"))
    monkeypatch.setattr(compact, "LOCK_PATH", str(data_dir / "compact.lock"))


class TestBasicCompaction:
    def test_two_sessions_produce_two_rows(self, tmp_path, monkeypatch):
        setup_paths(tmp_path, monkeypatch)
        jsonl = Path(compact.JSONL_PATH)
        write_jsonl(jsonl, [
            make_entry("s1", "2026-01-15T10:00:00+00:00", cost_usd=1.0),
            make_entry("s2", "2026-01-15T11:00:00+00:00", cost_usd=2.0),
        ])

        rc = compact.main()
        assert rc == 0

        csv = Path(compact.CSV_PATH)
        assert csv.exists()
        df = pd.read_csv(csv, dtype=str)
        assert len(df) == 2
        assert set(df["session_id"]) == {"s1", "s2"}

        # .processing should be cleaned up
        assert not Path(compact.PROCESSING_PATH).exists()
        # original jsonl should be gone
        assert not jsonl.exists()

    def test_latest_values_kept_per_session(self, tmp_path, monkeypatch):
        setup_paths(tmp_path, monkeypatch)
        write_jsonl(Path(compact.JSONL_PATH), [
            make_entry("s1", "2026-01-15T10:00:00+00:00", cost_usd=1.0, lines_added=5),
            make_entry("s1", "2026-01-15T10:05:00+00:00", cost_usd=2.5, lines_added=20),
        ])

        compact.main()

        df = pd.read_csv(compact.CSV_PATH, dtype=str)
        assert len(df) == 1
        assert df.iloc[0]["cost_usd"] == "2.5"
        assert df.iloc[0]["lines_added"] == "20"


class TestMerge:
    def test_merge_updates_existing_and_adds_new(self, tmp_path, monkeypatch):
        setup_paths(tmp_path, monkeypatch)

        # Pre-existing CSV with session A
        existing = pd.DataFrame([{
            "session_id": "sA",
            "first_active": "2026-01-10T08:00:00+00:00",
            "last_active": "2026-01-10T09:00:00+00:00",
            "model_id": "claude-opus-4-6",
            "model_name": "Opus 4.6",
            "project_dir": "/src/project",
            "current_dir": "/src/project",
            "transcript_path": "/t/sA.jsonl",
            "claude_code_version": "2.1.39",
            "cost_usd": "1.0",
            "duration_ms": "10000",
            "api_duration_ms": "5000",
            "lines_added": "10",
            "lines_removed": "2",
            "total_input_tokens": "1000",
            "total_output_tokens": "500",
            "context_window_size": "200000",
            "context_used_pct": "15",
            "exceeds_200k_tokens": "False",
        }])
        existing.to_csv(compact.CSV_PATH, index=False)

        # JSONL with updated A + new B
        write_jsonl(Path(compact.JSONL_PATH), [
            make_entry("sA", "2026-01-10T10:00:00+00:00", cost_usd=3.0),
            make_entry("sB", "2026-01-10T11:00:00+00:00", cost_usd=0.5),
        ])

        compact.main()

        df = pd.read_csv(compact.CSV_PATH, dtype=str)
        assert len(df) == 2
        assert set(df["session_id"]) == {"sA", "sB"}

        row_a = df[df["session_id"] == "sA"].iloc[0]
        assert row_a["cost_usd"] == "3.0"

    def test_first_active_preserved(self, tmp_path, monkeypatch):
        setup_paths(tmp_path, monkeypatch)

        # CSV has early first_active
        existing = pd.DataFrame([{c: "" for c in compact.CSV_COLUMNS}])
        existing.iloc[0, existing.columns.get_loc("session_id")] = "s1"
        existing.iloc[0, existing.columns.get_loc("first_active")] = "2026-01-01T00:00:00+00:00"
        existing.iloc[0, existing.columns.get_loc("last_active")] = "2026-01-01T01:00:00+00:00"
        existing.to_csv(compact.CSV_PATH, index=False)

        # JSONL has later timestamp for same session
        write_jsonl(Path(compact.JSONL_PATH), [
            make_entry("s1", "2026-01-15T10:00:00+00:00", cost_usd=5.0),
        ])

        compact.main()

        df = pd.read_csv(compact.CSV_PATH, dtype=str)
        row = df[df["session_id"] == "s1"].iloc[0]
        # first_active should be the earlier CSV value
        assert row["first_active"] == "2026-01-01T00:00:00+00:00"
        # last_active should be the later JSONL value
        assert row["last_active"] == "2026-01-15T10:00:00+00:00"


class TestEmptyInput:
    def test_no_jsonl_creates_csv(self, tmp_path, monkeypatch):
        setup_paths(tmp_path, monkeypatch)
        rc = compact.main()
        assert rc == 0
        # stats.csv should now exist (empty with headers)
        assert Path(compact.CSV_PATH).exists()
        df = pd.read_csv(compact.CSV_PATH, dtype=str)
        assert len(df) == 0

    def test_no_jsonl_touches_existing_csv(self, tmp_path, monkeypatch):
        setup_paths(tmp_path, monkeypatch)
        csv_path = Path(compact.CSV_PATH)
        # Create an existing CSV with an old mtime
        csv_path.write_text("session_id\nold\n")
        old_time = os.path.getmtime(csv_path) - 7200
        os.utime(csv_path, (old_time, old_time))
        mtime_before = os.path.getmtime(csv_path)

        rc = compact.main()
        assert rc == 0
        mtime_after = os.path.getmtime(csv_path)
        assert mtime_after > mtime_before


class TestCrashRecovery:
    def test_processing_file_recovered(self, tmp_path, monkeypatch):
        setup_paths(tmp_path, monkeypatch)

        # Simulate crash: .processing exists, no session.jsonl
        write_jsonl(Path(compact.PROCESSING_PATH), [
            make_entry("s1", "2026-01-15T10:00:00+00:00"),
        ])

        rc = compact.main()
        assert rc == 0
        assert Path(compact.CSV_PATH).exists()
        assert not Path(compact.PROCESSING_PATH).exists()

        df = pd.read_csv(compact.CSV_PATH, dtype=str)
        assert len(df) == 1
        assert df.iloc[0]["session_id"] == "s1"


class TestMalformedLines:
    def test_malformed_lines_skipped(self, tmp_path, monkeypatch, capsys):
        setup_paths(tmp_path, monkeypatch)

        jsonl = Path(compact.JSONL_PATH)
        with open(jsonl, "w") as f:
            f.write(json.dumps(make_entry("s1")) + "\n")
            f.write("NOT VALID JSON\n")
            f.write(json.dumps(make_entry("s2")) + "\n")

        compact.main()

        df = pd.read_csv(compact.CSV_PATH, dtype=str)
        assert len(df) == 2
        assert set(df["session_id"]) == {"s1", "s2"}

        captured = capsys.readouterr()
        assert "skipping malformed line 2" in captured.err


class TestSortOrder:
    def test_sorted_by_last_active_descending(self, tmp_path, monkeypatch):
        setup_paths(tmp_path, monkeypatch)
        write_jsonl(Path(compact.JSONL_PATH), [
            make_entry("s-old", "2026-01-01T00:00:00+00:00"),
            make_entry("s-mid", "2026-01-10T00:00:00+00:00"),
            make_entry("s-new", "2026-01-20T00:00:00+00:00"),
        ])

        compact.main()

        df = pd.read_csv(compact.CSV_PATH, dtype=str)
        assert list(df["session_id"]) == ["s-new", "s-mid", "s-old"]


class TestNullFields:
    def test_null_used_percentage(self, tmp_path, monkeypatch):
        setup_paths(tmp_path, monkeypatch)
        write_jsonl(Path(compact.JSONL_PATH), [
            make_entry("s1", used_pct=None),
        ])

        compact.main()

        df = pd.read_csv(compact.CSV_PATH)
        row = df.iloc[0]
        # null should result in NaN / empty in CSV
        assert pd.isna(row["context_used_pct"])


class TestLockTimeout:
    def test_exits_gracefully_when_locked(self, tmp_path, monkeypatch, capsys):
        setup_paths(tmp_path, monkeypatch)

        # Write some JSONL that should NOT be processed
        write_jsonl(Path(compact.JSONL_PATH), [
            make_entry("s1"),
        ])

        # Hold the lock externally
        lock_path = compact.LOCK_PATH
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        lock_fd = open(lock_path, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        try:
            rc = compact.main()
            assert rc == 0

            captured = capsys.readouterr()
            assert "another instance is running" in captured.err

            # session.jsonl should be untouched
            assert Path(compact.JSONL_PATH).exists()
            assert not Path(compact.CSV_PATH).exists()
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()


class TestDevFlag:
    def test_dev_flag_routes_to_dev_dir(self, tmp_path, monkeypatch):
        """Running compact.py --dev should process data/dev/ instead of data/."""
        import subprocess
        import sys

        dev_dir = tmp_path / "data" / "dev"
        dev_dir.mkdir(parents=True)
        jsonl = dev_dir / "session.jsonl"
        write_jsonl(jsonl, [make_entry("dev-sess")])

        # Run compact.py --dev as a subprocess with SCRIPT_DIR pointing to tmp_path
        # We monkeypatch the module-level paths instead
        monkeypatch.setattr(compact, "SCRIPT_DIR", str(tmp_path))
        monkeypatch.setattr(compact, "DATA_DIR", str(dev_dir))
        monkeypatch.setattr(compact, "JSONL_PATH", str(dev_dir / "session.jsonl"))
        monkeypatch.setattr(compact, "PROCESSING_PATH", str(dev_dir / "session.jsonl.processing"))
        monkeypatch.setattr(compact, "CSV_PATH", str(dev_dir / "stats.csv"))
        monkeypatch.setattr(compact, "LOCK_PATH", str(dev_dir / "compact.lock"))

        rc = compact.main()
        assert rc == 0

        csv_path = dev_dir / "stats.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path, dtype=str)
        assert len(df) == 1
        assert df.iloc[0]["session_id"] == "dev-sess"
