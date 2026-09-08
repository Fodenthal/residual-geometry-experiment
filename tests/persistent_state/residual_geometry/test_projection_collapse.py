from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from residual_geometry.residuals.probes import ResidualProbeSet


def _load_projection_collapse_module():
    script_dir = Path(__file__).parents[3] / "scripts" / "persistent_state" / "residual_geometry"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("projection_collapse", script_dir / "05_projection_collapse.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repairs_legacy_projection_probe_family_width() -> None:
    module = _load_projection_collapse_module()
    probes = ResidualProbeSet(
        directions=np.ones((2, 4), dtype=np.float32),
        probe_ids=np.array(["heldout_random_residual_00000", "heldout_time_lagged_residual_00000"]),
        probe_family=np.array(["r", "l"]),
    )

    repaired = module._repair_legacy_projection_probe_families(probes)

    assert repaired.probe_family.tolist() == ["random_heldout", "lag_heldout"]
