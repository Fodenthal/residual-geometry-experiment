#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 RUN_DIR SLOW_BASIS_NPZ" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$1"
SLOW_BASIS="$2"
PYTHON_BIN="${PYTHON:-python3}"
BATCH_SIZE="${SLOW_SEMANTIC_BATCH_SIZE:-8}"
STAGES="$REPO_ROOT/scripts/persistent_state/residual_geometry"
PROTOCOL="$REPO_ROOT/docs/protocols/slow-subspace-semantic-audit-r1.md"

cd "$REPO_ROOT"

"$PYTHON_BIN" "$STAGES/54_initialize_slow_semantic_audit.py" \
  --run-dir "$RUN_DIR" \
  --slow-basis "$SLOW_BASIS" \
  --spec "$PROTOCOL"
"$PYTHON_BIN" "$STAGES/55_prepare_slow_semantic_data.py" \
  --run-dir "$RUN_DIR" --batch-size "$BATCH_SIZE"
"$PYTHON_BIN" "$STAGES/56_capture_slow_semantic_residuals.py" \
  --run-dir "$RUN_DIR" --batch-size "$BATCH_SIZE"
"$PYTHON_BIN" "$STAGES/57_analyze_slow_semantic_profile.py" --run-dir "$RUN_DIR"
"$PYTHON_BIN" "$STAGES/58_analyze_slow_semantic_geometry.py" --run-dir "$RUN_DIR"
"$PYTHON_BIN" "$STAGES/59_make_slow_semantic_report.py" --run-dir "$RUN_DIR"
