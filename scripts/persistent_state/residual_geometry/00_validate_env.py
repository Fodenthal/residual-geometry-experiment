#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

from _common import artifact_metadata, load_config_and_store


def check(label: str, ok: bool, detail: str = "") -> bool:
    tag = "[PASS]" if ok else "[FAIL]"
    suffix = f" - {detail}" if detail else ""
    print(f"{tag} {label}{suffix}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate residual-geometry environment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-model-load", action="store_true")
    args = parser.parse_args()

    all_ok = True
    all_ok &= check("Python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0])
    for lib in ["torch", "transformer_lens", "datasets", "yaml", "pandas", "numpy", "sklearn"]:
        try:
            __import__(lib)
            all_ok &= check(f"import {lib}", True)
        except Exception as exc:
            all_ok &= check(f"import {lib}", False, str(exc))

    try:
        config, store = load_config_and_store(args.config)
        all_ok &= check("Load config", True, f"mode={config.mode}, run_name={config.run_name}")
        os.makedirs(config.output_base, exist_ok=True)
        all_ok &= check("Output base writable", True, os.path.abspath(config.output_base))
    except Exception as exc:
        all_ok &= check("Load config", False, str(exc))
        return 1

    if not args.skip_model_load:
        try:
            from residual_geometry.residuals.provider import ResidualStreamProvider
            from residual_geometry.utils.io import save_json

            provider = ResidualStreamProvider(config)
            provider.load()
            metadata = provider.architecture_metadata()
            metadata.update(artifact_metadata(config, "residual_geometry_00_validate_env"))
            save_json(metadata, store.resolved_model_architecture_path)
            all_ok &= check("Model load and architecture record", True, store.resolved_model_architecture_path)
        except Exception as exc:
            all_ok &= check("Model load and architecture record", False, str(exc))

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
