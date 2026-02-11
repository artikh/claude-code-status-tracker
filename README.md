# claude-code-status-tracker

A status line and usage tracker for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI. Shows real-time session metrics in the status bar and tracks usage across billing periods.

> **Disclaimer:** This is a personal "works on my machine" tool, built almost entirely with LLM assistance (Claude). It scratches my own itch and is shared as-is. Code quality, edge case handling, and portability are not guaranteed. Use at your own risk, steal what's useful, and don't expect production-grade polish.

## What it does

**Status line** — displays live session info at the bottom of your Claude Code terminal:

```
[Opus] ███░░░░░░░ 30% | +156 -23 lines | Stage 2: Core Logic | $0.45
```

- Model name, context window usage bar (color-coded), lines changed, project stage, session cost

**Usage tracking** — logs every session and provides analytics:

```bash
just sessions     # latest sessions table (most recent first)
just usage        # terminal report: today, this week, billing period, by workspace
just report       # PDF report for finished billing periods
```

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [just](https://github.com/casey/just) (command runner)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI

## Setup

```bash
git clone git@github.com:artikh/claude-code-status-tracker.git
cd claude-code-status-tracker
bash install.sh
```

`install.sh` checks prerequisites and then runs `just install`, which handles everything else: Python dependencies, git hooks, data directories, subscription config (interactive prompt), statusline wrapper, and validation.

After install, add to your `~/.claude/settings.json`:

```json
{
  "env": {
    "statusLine": "~/.claude/statusline.sh"
  }
}
```

## Usage

| Command | Description |
|---------|-------------|
| `just sessions` | Latest sessions table (last 20, `--all` for all) |
| `just usage` | Terminal usage report (current billing period) |
| `just report` | PDF report for finished billing periods |
| `just compact` | Compact session logs into stats CSV |
| `just check` | Run type checker + tests |
| `just install` | Re-run setup (skips prerequisite checks) |

## How it works

Claude Code pipes JSON session data to the status line script via stdin after each assistant turn. The script extracts fields (model, context usage, cost, lines changed) and prints ANSI-formatted text to stdout, which Claude Code renders in the status bar.

Session data is also logged to `data/session.jsonl` and periodically compacted into `data/stats.csv` for the analytics layer.

Subscription billing periods are configured in `data/settings.json` and support multi-currency with automatic USD exchange rate conversion. To renew a subscription, run `/renew-subscription` inside a Claude Code session in this project — it's a [Claude Code skill](https://docs.anthropic.com/en/docs/claude-code/skills) that computes the next billing dates, fetches the current exchange rate, and appends a new entry to `data/settings.json`.

## Project structure

```
track-and-status.py   # Status line script (entry point)
compact.py            # Session log compaction
sessions.py           # Latest sessions table
usage.py              # Terminal usage analytics
report_pdf.py         # PDF report generator
install.sh            # Prerequisite checker
justfile              # Task runner recipes
data/settings.json    # Subscription & timezone config
data/session.jsonl    # Raw session log (gitignored)
data/stats.csv        # Compacted stats (gitignored)
```

## License

[CC0 1.0 Universal](LICENSE) — public domain. Do whatever you want with it.
