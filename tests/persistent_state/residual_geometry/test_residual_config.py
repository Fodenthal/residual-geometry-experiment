from __future__ import annotations

from pathlib import Path

from residual_geometry.config.loader import load_persistent_state_config


def test_residual_geometry_smoke_config_uses_1024_token_window() -> None:
    config = load_persistent_state_config("configs/persistent_state/residual_geometry/smoke.yaml")
    assert config.context_processing.max_tokens == 1024
    assert config.context_processing.add_special_tokens is False
    assert config.residual_geometry is not None
    assert config.residual_geometry.max_tokens == 1024
    assert config.residual_geometry.max_lag == 128
    assert config.residual_geometry.random.directions == 64
    assert config.residual_geometry.valid_doc_threshold == 8


def test_residual_geometry_full_allows_768_sensitivity_ci_lag() -> None:
    config = load_persistent_state_config("configs/persistent_state/residual_geometry/full.yaml")
    assert config.residual_geometry is not None
    assert config.residual_geometry.max_lag == 512
    assert config.residual_geometry.sensitivity_max_lag == 768
    assert 768 in config.residual_geometry.ci_lags


def test_residual_scripts_do_not_use_legacy_autocorr_max_lag() -> None:
    script_dir = Path("scripts/persistent_state/residual_geometry")
    offenders = []
    for path in script_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "config.autocorr.max_lag" in text or "cfg.autocorr.max_lag" in text:
            offenders.append(str(path))
    assert offenders == []
