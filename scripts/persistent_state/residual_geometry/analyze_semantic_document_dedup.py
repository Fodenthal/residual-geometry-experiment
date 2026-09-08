#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_config_and_store
from semantic_axis_comparison import (
    _document_level_span_metrics,
    _locked_span_samples,
    _saved_projection_bundle,
    _selection_table,
    _spans,
    _top_entries,
)


def _group_summary(direction_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        direction_summary.groupby("selection_reason")
        .agg(
            n_directions=("probe_id", "count"),
            median_tau=("tau_within", "median"),
            median_q95_span_count=("n_retained_q95_spans", "median"),
            median_docs_with_retained_q95_span=("n_docs_with_retained_q95_span", "median"),
            q10_docs_with_retained_q95_span=("n_docs_with_retained_q95_span", lambda values: values.quantile(0.10)),
            median_fraction_docs_with_retained_q95_span=("fraction_docs_with_retained_q95_span", "median"),
            q10_fraction_docs_with_retained_q95_span=(
                "fraction_docs_with_retained_q95_span",
                lambda values: values.quantile(0.10),
            ),
            median_spans_per_covered_doc=("spans_per_covered_doc", "median"),
            median_doc_max_q95_span_length_covered=("median_doc_max_q95_span_length_covered", "median"),
            median_mean_doc_max_q95_span_length_all_docs=("mean_doc_max_q95_span_length_all_docs", "median"),
            median_unique_top_window_docs=("n_unique_top_window_docs", "median"),
            median_max_top_window_doc_share=("max_top_window_doc_share", "median"),
        )
        .reset_index()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recount semantic-axis q95 spans after collapsing repeated spans and windows to distinct documents."
    )
    parser.add_argument("--config", default="configs/persistent_state/residual_geometry/pilot.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--random-count", type=int, default=10)
    parser.add_argument("--span-sample-count", type=int, default=10)
    parser.add_argument("--tail-windows", type=int, default=12)
    parser.add_argument("--min-span-length", type=int, default=8)
    parser.add_argument("--max-documents", type=int, default=None, help="Limit validation documents for benchmarking.")
    args = parser.parse_args()

    _, store = load_config_and_store(args.config)
    out_dir = Path(args.out_dir or Path(store.subspace_dir) / "semantic_axis_document_dedup")
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = _selection_table(store, random_count=args.random_count)
    span_selection, derived_groups = _locked_span_samples(store, count=args.span_sample_count)
    selection = pd.concat([selection, span_selection], ignore_index=True)
    _, _, projected = _saved_projection_bundle(
        store,
        split="val",
        probe_ids=selection["probe_id"].astype(str).to_numpy(),
        max_documents=args.max_documents,
        derived_groups=derived_groups,
    )

    rows: list[dict[str, object]] = []
    for probe_idx, selected in selection.reset_index(drop=True).iterrows():
        values = projected[:, :, probe_idx].astype(np.float32)
        threshold = float(np.quantile(values, 0.95))
        retained = _spans(values, threshold=threshold, min_length=args.min_span_length)
        top_entries = _top_entries(values, args.tail_windows, largest=True)
        bottom_entries = _top_entries(values, args.tail_windows, largest=False)
        rows.append(
            {
                **selected.to_dict(),
                "q95_threshold": threshold,
                "min_span_length": int(args.min_span_length),
                "n_retained_q95_spans": int(len(retained)),
                "mean_retained_q95_span_length": (
                    float(np.mean([stop - start for _, start, stop, _, _ in retained])) if retained else 0.0
                ),
                **_document_level_span_metrics(
                    retained,
                    n_documents=len(values),
                    top_entries=top_entries,
                    bottom_entries=bottom_entries,
                ),
            }
        )

    direction_summary = pd.DataFrame(rows)
    group_summary = _group_summary(direction_summary)
    direction_path = out_dir / "document_dedup_direction_summary.csv"
    group_path = out_dir / "document_dedup_group_summary.csv"
    direction_summary.to_csv(direction_path, index=False)
    group_summary.to_csv(group_path, index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "projection_data_source": "saved_stage_b_validation_projection_chunks",
                "validation_documents": int(projected.shape[0]),
                "directions": int(projected.shape[-1]),
                "tail_windows_per_polarity": int(args.tail_windows),
                "q95_threshold_scope": "global token-level projection distribution within each direction",
                "minimum_retained_span_length": int(args.min_span_length),
                "direction_summary_path": str(direction_path),
                "group_summary_path": str(group_path),
                "notes": [
                    "Document-level metrics count each validation document at most once when asking whether a direction has a retained q95 span.",
                    "Per-document maximum span metrics prevent multiple contiguous runs in one page from inflating the long-span comparison.",
                    "Top-window document concentration is reported separately because document-state directions are expected to produce correlated windows within a page.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(group_summary.to_string(index=False))
    print(f"\nWrote document-deduplicated semantic comparison to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
