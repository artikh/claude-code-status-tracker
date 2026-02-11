#!/usr/bin/env bash
set -euo pipefail

# Read hook JSON from stdin, check if a .py file was edited
input=$(cat)
file_path=$(echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null || true)

if [[ "$file_path" == *.py ]]; then
    cd "$(dirname "$0")/../.."
    just test
fi
