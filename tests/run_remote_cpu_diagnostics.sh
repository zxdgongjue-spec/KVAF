#!/usr/bin/env bash
set -euo pipefail
KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$KVAF_ROOT/tests/diagnose_remote_cpu.py" "$@"
