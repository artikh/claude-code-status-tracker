# CLAUDE.md

## Project Identity

**claude-code-status-tracker** is a Python-based status line and session tracker for Claude Code CLI. It displays real-time session metrics in the Claude Code status bar and is designed to be extended with persistent session tracking.

- **Language:** Python 3.13+
- **Package manager:** uv
- **Entry point:** `track-and-status.py`

## How It Works

Claude Code pipes JSON session data to the status line script via stdin after each assistant message (debounced at 300ms). The script reads the JSON, extracts fields, and prints formatted text to stdout. Claude Code renders whatever the script prints in the status bar at the bottom of the terminal.

### Deployment

Claude Code settings (`~/.claude/settings.json`) point to `~/.claude/statusline.sh`, which is a wrapper script (not a symlink) that sets `COLLECT_USAGE=prod` so production data is written to `data/session.jsonl`:

```bash
#!/bin/bash
COLLECT_USAGE=prod exec /src/claude-code-status-tracker/track-and-status.py
```

Without `COLLECT_USAGE=prod` (e.g. manual `echo '...' | python3 track-and-status.py`), data goes to `data/dev/session.jsonl` to avoid polluting production logs.

### Current Status Line Output

```
[Opus] █░░░░░░░░░ 17% | +156 -23 lines | Stage 1: Scaffolding | $0.45
```

Components:
- **Model name** in cyan
- **Context bar** (10-char) — green < 70%, yellow 70–89%, red 90%+
- **Lines added/removed** — green/red
- **Project stage** — parsed from the workspace's CLAUDE.md `## Current Stage` section, cached 30s
- **Session cost** in yellow

ANSI escape codes work in the status line (but only via script files — inline JSON commands double-escape them).

## Available JSON Fields

These fields are sent by Claude Code on stdin:

| Field | Description |
|-------|-------------|
| `session_id` | Unique session identifier |
| `model.id`, `model.display_name` | Model info |
| `workspace.current_dir` | Current working directory |
| `workspace.project_dir` | Directory where Claude Code was launched |
| `cost.total_cost_usd` | Session cost in USD (likely includes subagent costs) |
| `cost.total_duration_ms` | Wall-clock time since session start |
| `cost.total_api_duration_ms` | Time waiting for API responses |
| `cost.total_lines_added` | Lines of code added this session |
| `cost.total_lines_removed` | Lines of code removed this session |
| `context_window.used_percentage` | Context window usage % |
| `context_window.remaining_percentage` | Context window remaining % |
| `context_window.context_window_size` | Max context (200k or 1M tokens) |
| `context_window.total_input_tokens` | Cumulative input tokens |
| `context_window.total_output_tokens` | Cumulative output tokens |
| `transcript_path` | Path to conversation transcript file |
| `version` | Claude Code version |
| `output_style.name` | Current output style |
| `vim.mode` | Vim mode if enabled (NORMAL/INSERT) |
| `agent.name` | Agent name if using --agent |

**Note on subagent costs:** `cost.total_cost_usd` tracks total session cost. Subagents (Task tool) run within the same session and their API costs likely roll up to this total. Whether `total_lines_added/removed` includes subagent edits is unconfirmed.

## Testing

```bash
# Test with mock JSON input
echo '{"model":{"display_name":"Opus"},"context_window":{"used_percentage":17},"cost":{"total_cost_usd":0.45,"total_lines_added":156,"total_lines_removed":23},"workspace":{"current_dir":"/src/langcardhelper"},"session_id":"abc123"}' | python3 track-and-status.py
```

## Planned Features

- Persistent session tracking using `session_id` (scope TBD)

## Build & Run

```bash
# No dependencies yet — runs with stdlib only
python3 track-and-status.py < mock-input.json

# When dependencies are added
uv run track-and-status.py < mock-input.json
```
