#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" == "--full" ]]; then
  python reproduce.py --full
else
  python reproduce.py --quick
fi
