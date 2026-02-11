# Repo root (used by install recipes)
_repo := justfile_directory()

# Run unit tests
test:
    uv run pytest tests/ -v

# Compact session log into stats CSV
compact:
    uv run compact.py

# Compact dev session log into dev stats CSV
compact-dev:
    uv run compact.py --dev

# Show usage stats from production data
usage:
    uv run usage.py

# Show usage stats from dev data
usage-dev:
    uv run usage.py --dev

# Show latest sessions from production data
sessions:
    uv run sessions.py

# Show latest sessions from dev data
sessions-dev:
    uv run sessions.py --dev

# Generate PDF report from production data
report:
    uv run report_pdf.py

# Generate PDF report from dev data
report-dev:
    uv run report_pdf.py --dev

# Run type checker
typecheck:
    uv run mypy track-and-status.py compact.py usage.py sessions.py report_pdf.py tests/

# Run all checks
check: typecheck test

# Configure git to use .githooks/ directory
setup:
    git config core.hooksPath .githooks

# Run setup & validation (use install.sh for prerequisite checks first)
install: _uv-sync _setup-hooks _ensure-dirs _ensure-settings _ensure-subscription _ensure-statusline check
    @echo -e '\n\033[0;32mAll checks passed.\033[0m'

_uv-sync:
    #!/usr/bin/env bash
    set -euo pipefail
    G='\033[0;32m' R='\033[0;31m' C='\033[0;36m' N='\033[0m'
    echo -e "\n${C}Python dependencies${N}"
    if uv sync --quiet 2>/dev/null; then
        echo -e "  ${G}✔${N} uv sync"
    else
        echo -e "  ${R}✘${N} uv sync failed"
        exit 1
    fi

_setup-hooks:
    #!/usr/bin/env bash
    set -euo pipefail
    G='\033[0;32m' Y='\033[0;33m' R='\033[0;31m' C='\033[0;36m' N='\033[0m'
    echo -e "\n${C}Git hooks${N}"
    HOOKS_PATH=$(git config core.hooksPath 2>/dev/null || echo "")
    if [[ "$HOOKS_PATH" == ".githooks" ]]; then
        echo -e "  ${G}✔${N} core.hooksPath = .githooks"
    else
        git config core.hooksPath .githooks
        echo -e "  ${Y}✔${N} core.hooksPath set to .githooks ${Y}(fixed)${N}"
    fi
    if [[ -x ".githooks/pre-commit" ]]; then
        echo -e "  ${G}✔${N} .githooks/pre-commit is executable"
    elif [[ -f ".githooks/pre-commit" ]]; then
        chmod +x .githooks/pre-commit
        echo -e "  ${Y}✔${N} .githooks/pre-commit made executable ${Y}(fixed)${N}"
    else
        echo -e "  ${R}✘${N} .githooks/pre-commit not found"
        exit 1
    fi

_ensure-dirs:
    #!/usr/bin/env bash
    set -euo pipefail
    G='\033[0;32m' Y='\033[0;33m' C='\033[0;36m' N='\033[0m'
    echo -e "\n${C}Data directories${N}"
    for dir in data data/dev; do
        if [[ -d "$dir" ]]; then
            echo -e "  ${G}✔${N} $dir/"
        else
            mkdir -p "$dir"
            echo -e "  ${Y}✔${N} $dir/ ${Y}(created)${N}"
        fi
    done

_ensure-settings:
    #!/usr/bin/env bash
    set -euo pipefail
    G='\033[0;32m' Y='\033[0;33m' C='\033[0;36m' N='\033[0m'
    echo -e "\n${C}Settings${N}"
    SETTINGS="data/settings.json"
    if [[ -f "$SETTINGS" ]]; then
        echo -e "  ${G}✔${N} $SETTINGS exists"
    else
        echo '{"subscriptions": [], "timezone": "UTC"}' > "$SETTINGS"
        echo -e "  ${Y}✔${N} $SETTINGS ${Y}(created)${N}"
    fi

_ensure-subscription:
    #!/usr/bin/env bash
    set -euo pipefail
    G='\033[0;32m' Y='\033[0;33m' C='\033[0;36m' N='\033[0m'
    echo -e "\n${C}Subscription${N}"
    SETTINGS="data/settings.json"
    HAS_ACTIVE=$(python3 -c "
    import json, datetime
    with open('$SETTINGS') as f:
        data = json.load(f)
    today = datetime.date.today().isoformat()
    active = any(s['start'] <= today <= s['end'] for s in data.get('subscriptions', []))
    print(int(active))
    " 2>/dev/null || echo 0)
    if [[ "$HAS_ACTIVE" == "1" ]]; then
        echo -e "  ${G}✔${N} active subscription found"
        exit 0
    fi
    echo -e "  ${C}ℹ${N} no active subscription — let's add one"
    echo ""
    DEF_PLAN="Claude Max 20x"
    DEF_COST="200"
    DEF_CURRENCY="USD"
    DEF_START=$(python3 -c "import datetime; d=datetime.date.today(); print(d.replace(day=1).isoformat())")
    DEF_END=$(python3 -c "
    import datetime, calendar
    d = datetime.date.today()
    last = calendar.monthrange(d.year, d.month)[1]
    print(d.replace(day=last).isoformat())
    ")
    read -rp "  Plan name [$DEF_PLAN]: " PLAN
    PLAN=${PLAN:-$DEF_PLAN}
    read -rp "  Cost [$DEF_COST]: " COST
    COST=${COST:-$DEF_COST}
    read -rp "  Currency [$DEF_CURRENCY]: " CURRENCY
    CURRENCY=${CURRENCY:-$DEF_CURRENCY}
    CURRENCY=$(echo "$CURRENCY" | tr '[:lower:]' '[:upper:]')
    read -rp "  Start date [$DEF_START]: " START
    START=${START:-$DEF_START}
    read -rp "  End date [$DEF_END]: " END
    END=${END:-$DEF_END}
    USD_RATE="1.0"
    if [[ "$CURRENCY" != "USD" ]]; then
        echo -e "  ${C}ℹ${N} fetching exchange rate for $CURRENCY..."
        RATE_JSON=$(curl -sf "https://open.er-api.com/v6/latest/USD" 2>/dev/null || echo "")
        if [[ -n "$RATE_JSON" ]]; then
            USD_RATE=$(echo "$RATE_JSON" | python3 -c "
    import json, sys
    data = json.loads(sys.stdin.read())
    rate = data.get('rates', {}).get('$CURRENCY')
    if rate and rate > 0:
        print(round(1.0 / rate, 8))
    else:
        print('')
    " 2>/dev/null || echo "")
        fi
        if [[ -z "$USD_RATE" ]]; then
            echo -e "  ${C}ℹ${N} could not fetch rate automatically"
            read -rp "  USD rate (1 $CURRENCY = ? USD): " USD_RATE
        else
            echo -e "  ${C}ℹ${N} 1 $CURRENCY = $USD_RATE USD"
        fi
    fi
    python3 -c "
    import json
    with open('$SETTINGS') as f:
        data = json.load(f)
    data.setdefault('subscriptions', []).append({
        'plan': '$PLAN',
        'cost': $COST,
        'currency': '$CURRENCY',
        'usd_rate': $USD_RATE,
        'start': '$START',
        'end': '$END',
    })
    with open('$SETTINGS', 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    "
    echo -e "  ${Y}✔${N} subscription added to $SETTINGS ${Y}(fixed)${N}"

_ensure-statusline:
    #!/usr/bin/env bash
    set -euo pipefail
    G='\033[0;32m' Y='\033[0;33m' C='\033[0;36m' N='\033[0m'
    echo -e "\n${C}Statusline wrapper${N}"
    REPO="{{_repo}}"
    WRAPPER="$HOME/.claude/statusline.sh"
    printf -v EXPECTED '#!/bin/bash\nCOLLECT_USAGE=prod exec %s/track-and-status.py' "$REPO"
    if [[ -f "$WRAPPER" ]]; then
        CURRENT=$(cat "$WRAPPER")
        if [[ "$CURRENT" == "$EXPECTED" ]]; then
            echo -e "  ${G}✔${N} $WRAPPER up to date"
        else
            echo ""
            echo -e "  ${C}ℹ${N} content differs from expected:"
            diff <(echo "$CURRENT") <(echo "$EXPECTED") || true
            echo ""
            read -rp "  Overwrite $WRAPPER? [y/N]: " OVERWRITE
            if [[ "$OVERWRITE" =~ ^[Yy]$ ]]; then
                echo "$EXPECTED" > "$WRAPPER"
                chmod +x "$WRAPPER"
                echo -e "  ${Y}✔${N} $WRAPPER ${Y}(updated)${N}"
            else
                echo -e "  ${C}ℹ${N} skipped — leaving $WRAPPER unchanged"
            fi
        fi
    else
        mkdir -p "$(dirname "$WRAPPER")"
        echo "$EXPECTED" > "$WRAPPER"
        chmod +x "$WRAPPER"
        echo -e "  ${Y}✔${N} $WRAPPER ${Y}(created)${N}"
    fi
    CLAUDE_SETTINGS="$HOME/.claude/settings.json"
    if [[ -f "$CLAUDE_SETTINGS" ]]; then
        HAS_SL=$(python3 -c "
    import json
    with open('$CLAUDE_SETTINGS') as f:
        data = json.load(f)
    sl = data.get('env', {}).get('statusLine', '')
    print(int('statusline.sh' in sl))
    " 2>/dev/null || echo 0)
        if [[ "$HAS_SL" == "1" ]]; then
            echo -e "  ${G}✔${N} Claude Code settings reference statusline.sh"
        else
            echo -e "  ${C}ℹ${N} Claude Code settings (~/.claude/settings.json) may need statusLine config:"
            echo -e "  ${C}ℹ${N}   \"env\": { \"statusLine\": \"~/.claude/statusline.sh\" }"
        fi
    else
        echo -e "  ${C}ℹ${N} ~/.claude/settings.json not found — configure statusLine manually"
    fi
