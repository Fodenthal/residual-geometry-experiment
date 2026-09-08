#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    s0 = json.loads((run / "decisions" / "s0.json").read_text())
    s2 = json.loads((run / "decisions" / "s2_profile.json").read_text())
    s3_path = run / "decisions" / "s3_geometry.json"
    s3 = json.loads(s3_path.read_text()) if s3_path.exists() else {"status": "NOT_RUN"}
    table = pd.read_parquet(run / "supervised" / "bootstrap_summary.parquet")
    lines = [
        "# Slow-Subspace Semantic Audit R1 — Final Report", "",
        f"- S0: `{s0['status']}` / `{s0['slow_basis_status']}`",
        f"- S1: `{s0['s1_status']}`",
        f"- S2 profile: `{s2['profile']}`",
        f"- S3: `{s3['status']}`", "",
        "## Sealed-test document-bootstrap results", "",
        "| target | rank | statistic | point (nats) | 95% CI |", "|---|---:|---|---:|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(f"| {row.target} | {row.rank} | {row.statistic} | {row.point:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] |")
    lines += ["", "## Claim boundary", ""]
    if s2["profile"] == "PROFILE_GENERIC":
        lines.append("The slow space is temporally distinctive but has no distinctive semantic/document-state profile under this frozen assay.")
    elif s2["profile"] == "PROFILE_STRUCTURAL_DURABLE":
        lines.append("The original slow space is preferentially a durable structural/source-state carrier, not a clean semantic substrate.")
    elif s2["profile"] == "PROFILE_DURABLE_SEMANTIC":
        lines.append(
            "The frozen coarse-topic endpoint was statistically positive. This is moderate evidence "
            "for a semantic interpretation because topic may still be confounded with other durable "
            "document properties such as source or style."
        )
    else:
        lines.append("The slow space carries a mixture of semantic and nonsemantic durable document state.")
    lines += ["", "No causal claim is licensed by this run. S4 was not queued.", ""]
    (run / "reports" / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    (run / "reports" / "final_summary.json").write_text(json.dumps({"s0": s0, "s2": s2, "s3": s3}, indent=2) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
