#!/usr/bin/env python
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from residual_geometry.slow_semantic.protocol import (
    PROTOCOL,
    load_basis,
    protocol_hash,
    save_json,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze and audit the slow semantic experiment.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--slow-basis", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--software-commit")
    parser.add_argument("--allow-prospective-refit", action="store_true")
    args = parser.parse_args()

    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    for rel in (
        "provenance", "sae_bridge", "labels", "capture", "supervised",
        "semantic_geometry", "decisions", "reports", "logs",
    ):
        (run / rel).mkdir(exist_ok=True)

    spec = Path(args.spec).resolve()
    frozen_spec = run / "provenance" / "slow_subspace_semantic_audit_r1.md"
    shutil.copy2(spec, frozen_spec)
    basis_path = Path(args.slow_basis).resolve()
    if not basis_path.exists():
        if args.allow_prospective_refit:
            save_json(run / "decisions" / "s0.json", {
                "status": "PROSPECTIVE_REFIT_REQUIRED",
                "reason": f"original basis absent: {basis_path}",
            })
            return 3
        raise FileNotFoundError(basis_path)

    basis, array_key = load_basis(basis_path)
    digest = sha256_file(basis_path)
    if digest != PROTOCOL["original_basis_sha256"]:
        raise ValueError(
            f"Basis hash mismatch: got {digest}, expected {PROTOCOL['original_basis_sha256']}. "
            "A prospective refit must use a separate explicitly labeled run."
        )
    np.savez_compressed(
        run / "provenance" / "slow_basis_rank31.npz",
        directions=basis.astype(np.float32),
        source_array_key=np.array(array_key),
        source_sha256=np.array(digest),
    )
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    commit = args.software_commit or (git_result.stdout.strip() if git_result.returncode == 0 else None)
    if not commit:
        raise ValueError("software commit is required for a deployed source snapshot without .git metadata")
    manifest = {
        **PROTOCOL,
        "protocol_hash": protocol_hash(),
        "spec_sha256": sha256_file(frozen_spec),
        "software_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "slow_basis_status": "ORIGINAL_FROZEN",
        "slow_basis_source": str(basis_path),
        "slow_basis_sha256": digest,
        "slow_basis_array_key": array_key,
        "slow_basis_orthonormality_spectral_error": float(
            np.linalg.norm(basis.T @ basis - np.eye(31), ord=2)
        ),
        "label_blinding": "labels must complete before capture/analysis",
        "test_access": "single access by stage 57 after procedure freeze",
    }
    save_json(run / "provenance" / "protocol.json", manifest)
    save_json(run / "sae_bridge" / "status.json", {
        "status": "SAE_BRIDGE_NOT_EVALUATED",
        "reason": PROTOCOL["sae_policy"],
        "s2_authorized": True,
    })
    save_json(run / "decisions" / "s0.json", {
        "status": "PASS",
        "slow_basis_status": "ORIGINAL_FROZEN",
        "s1_status": "SAE_BRIDGE_NOT_EVALUATED",
        "s2_authorized": True,
    })
    print(f"initialized {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
