#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
CONFIG="configs/persistent_state/residual_geometry/${MODE}.yaml"
SCRIPT_DIR="scripts/persistent_state/residual_geometry"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Unknown mode '${MODE}'. Expected one of: smoke, pilot, full." >&2
  exit 2
fi

python "${SCRIPT_DIR}/00_validate_env.py" --config "${CONFIG}"
python "${SCRIPT_DIR}/01_build_context_pool.py" --config "${CONFIG}"
python "${SCRIPT_DIR}/02_compute_residual_probes.py" --config "${CONFIG}"
python "${SCRIPT_DIR}/03_compute_residual_autocorr.py" --config "${CONFIG}"
python "${SCRIPT_DIR}/04_residual_subspace_pilot.py" --config "${CONFIG}"
python "${SCRIPT_DIR}/05_projection_collapse.py" --config "${CONFIG}"
python "${SCRIPT_DIR}/05b_fat_subspace_diagnostics.py" --config "${CONFIG}"
python "${SCRIPT_DIR}/06_attention_alignment.py" --config "${CONFIG}"
python "${SCRIPT_DIR}/07_make_residual_report.py" --config "${CONFIG}"
