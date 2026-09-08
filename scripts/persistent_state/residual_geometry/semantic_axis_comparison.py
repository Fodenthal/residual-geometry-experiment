#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from _common import load_config_and_store, require_residual_geometry_config, residual_batches, split_contexts, token_matrix
from residual_geometry.residuals.probes import ResidualProbeSet, load_many_probe_sets
from residual_geometry.residuals.projections import load_projection_chunk, project_residuals


@dataclass(frozen=True)
class DerivedProjectionGroup:
    target_probe_ids: np.ndarray
    source_probe_ids: np.ndarray
    coefficients: np.ndarray


LABEL_PATTERNS = {
    "technical_software": r"\b(software|server|network|device|configuration|install|database|protocol|camera|code|python|windows)\b",
    "legal_privacy_regulatory": r"\b(privacy|terms|policy|agreement|liability|regulation|gdpr|statute|court|legal|license)\b",
    "product_catalog_seo": r"\b(product|buy|price|sale|supplier|catalog|shipping|discount|order|shop|customer)\b",
    "numeric_table_units": r"\b\d+(?:\.\d+)?\s?(?:%|kg|cm|mm|mg|ml|gb|mb|hz|rpm|°c|usd|dollars?)\b",
    "biomedical_science": r"\b(patient|clinical|study|medical|disease|treatment|surgery|protein|cells?|therapy|diagnosis)\b",
    "academic_policy_institutional": r"\b(university|research|government|committee|education|institution|report|department|policy|study)\b",
    "recipe_food_lifestyle": r"\b(recipe|cook|bake|ingredients?|cup|tablespoon|food|menu|kitchen|diet)\b",
    "religious_devotional": r"\b(god|jesus|christ|church|bible|lord|faith|prayer|scripture|israel)\b",
    "sports_entertainment": r"\b(game|team|player|season|film|movie|music|sports?|football|baseball|celebrity)\b",
    "code_changelog": r"\b(version|release|patch|commit|changelog|function|class|github|api|bug)\b",
    "scraped_noise_spam": r"\b(click here|free download|make money|seo|casino|dating|viagra|admin|followers)\b",
    "has_url_or_html": r"(https?://|www\.|</?[a-z][^>]*>)",
    "question_answer_or_forum": r"\b(question|answer|forum|reply|posted by|comment|asked|answered)\b",
}


def _load_probe_table(store) -> pd.DataFrame:
    path = Path(store.residual_probe_timescales_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run 03_compute_residual_autocorr.py first.")
    df = pd.read_parquet(path)
    return df[(df["split"] == "val") & (df["control"] == "real")].copy()


def _selection_table(store, random_count: int) -> pd.DataFrame:
    timescales = _load_probe_table(store).set_index("probe_id")
    with np.load(store.residual_pca_directions_path, allow_pickle=False) as pca:
        pca_ids = pca["probe_ids"].astype(str)
        explained = pca["explained_variance_ratio"].astype(float)

    reasons: dict[str, set[str]] = defaultdict(set)
    for probe_id in pca_ids[:10]:
        reasons[probe_id].add("pca_top_variance")
    pca_times = timescales.loc[pca_ids].sort_values(["tau_within", "probe_id"], ascending=[False, True])
    for probe_id in pca_times.head(10).index:
        reasons[probe_id].add("pca_top_lifetime")
    tl = timescales[timescales["probe_family"] == "time_lagged"].sort_values(
        ["tau_within", "probe_id"], ascending=[False, True]
    )
    for probe_id in tl.head(10).index:
        reasons[probe_id].add("persistent_reference")
    random = timescales[timescales["probe_family"] == "random"].sort_index()
    sampled_random = random.sample(n=min(random_count, len(random)), random_state=20260530)
    for probe_id in sampled_random.index:
        reasons[probe_id].add("random_control")

    explained_by_id = dict(zip(pca_ids, explained))
    rows = []
    for probe_id, probe_reasons in sorted(reasons.items()):
        row = timescales.loc[probe_id]
        rows.append(
            {
                "probe_id": probe_id,
                "probe_family": row["probe_family"],
                "selection_reason": "+".join(sorted(probe_reasons)),
                "tau_within": float(row["tau_within"]),
                "explained_variance_ratio": explained_by_id.get(probe_id, np.nan),
            }
        )
    return pd.DataFrame(rows)


def _selected_probes(store, selection: pd.DataFrame) -> ResidualProbeSet:
    all_probes = load_many_probe_sets(
        [
            store.time_lagged_residual_directions_path,
            store.residual_pca_directions_path,
            store.random_residual_directions_path,
        ]
    )
    index = {probe_id: idx for idx, probe_id in enumerate(all_probes.probe_ids.astype(str))}
    selected = [index[probe_id] for probe_id in selection["probe_id"]]
    return ResidualProbeSet(
        directions=all_probes.directions[selected],
        probe_ids=all_probes.probe_ids[selected],
        probe_family=all_probes.probe_family[selected],
    )


def _derived_projection_group(
    target_probe_ids: np.ndarray,
    target_directions: np.ndarray,
    source_probe_ids: np.ndarray,
    source_directions: np.ndarray,
) -> DerivedProjectionGroup:
    coefficients = np.linalg.lstsq(
        source_directions.astype(np.float64).T,
        target_directions.astype(np.float64).T,
        rcond=None,
    )[0].T
    reconstruction = coefficients @ source_directions
    max_error = float(np.max(np.linalg.norm(reconstruction - target_directions, axis=1)))
    if max_error > 1e-4:
        raise ValueError(f"Could not reconstruct locked span directions from saved parent axes; max_error={max_error:.3g}")
    return DerivedProjectionGroup(
        target_probe_ids=target_probe_ids.astype(str),
        source_probe_ids=source_probe_ids.astype(str),
        coefficients=coefficients.astype(np.float32),
    )


def _locked_span_samples(store, count: int) -> tuple[pd.DataFrame, list[DerivedProjectionGroup]]:
    if count <= 0:
        raise ValueError("--span-sample-count must be positive")
    fat_dir = Path(store.fat_subspace_dir)
    timescale_path = fat_dir / "direction_timescales.parquet"
    headline_path = Path(store.subspace_dir) / "projection_bases" / "residual_first_kstar.npz"
    endpoint_path = fat_dir / "random_in_span_directions.npz"
    pca_span_path = fat_dir / "direct_random_pca_span_directions.npz"
    for path in [timescale_path, headline_path, endpoint_path, pca_span_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run 05b_fat_subspace_diagnostics.py first.")

    all_probes = load_many_probe_sets(
        [
            store.time_lagged_residual_directions_path,
            store.residual_pca_directions_path,
            store.random_residual_directions_path,
        ]
    )
    direction_by_id = {
        str(probe_id): direction
        for probe_id, direction in zip(all_probes.probe_ids.astype(str), all_probes.directions)
    }
    with np.load(store.residual_pca_directions_path, allow_pickle=False) as pca:
        pca_ids = pca["probe_ids"].astype(str)
        pca_directions = pca["directions"].astype(np.float32)
    with np.load(headline_path, allow_pickle=False) as headline:
        source_ids = headline["source_probe_ids"].astype(str)
    source_directions = np.stack([direction_by_id[probe_id] for probe_id in source_ids])
    timescales = pd.read_parquet(timescale_path)
    timescales = timescales[(timescales["split"] == "val") & timescales["tau_valid_within"].astype(bool)].copy()
    tau_by_key = {
        (str(row.diagnostic_family), str(row.probe_id)): float(row.tau_within)
        for row in timescales.itertuples(index=False)
    }

    rows: list[dict[str, object]] = []
    groups: list[DerivedProjectionGroup] = []

    def add_group(path: Path, family: str, parent_ids: np.ndarray, parent_directions: np.ndarray) -> None:
        with np.load(path, allow_pickle=False) as data:
            families = data["probe_family"].astype(str)
            selected = np.flatnonzero(families == family)[:count]
            if len(selected) < count:
                raise ValueError(f"Locked artifact {path} contains only {len(selected)} directions for family={family!r}")
            target_ids = data["probe_ids"].astype(str)[selected]
            target_directions = data["directions"].astype(np.float32)[selected]
        groups.append(_derived_projection_group(target_ids, target_directions, parent_ids, parent_directions))
        for probe_id in target_ids:
            rows.append(
                {
                    "probe_id": str(probe_id),
                    "probe_family": family,
                    "selection_reason": f"{family}_locked_sample",
                    "tau_within": tau_by_key[(family, str(probe_id))],
                    "explained_variance_ratio": np.nan,
                }
            )

    with np.load(endpoint_path, allow_pickle=False) as endpoint:
        endpoint_families = endpoint["probe_family"].astype(str)
    endpoint_family = str(endpoint_families[0])
    if len(set(endpoint_families)) != 1 or not endpoint_family.startswith("random_in_q"):
        raise ValueError(f"Expected one random_in_q endpoint family in {endpoint_path}")
    add_group(endpoint_path, endpoint_family, source_ids, source_directions)
    for pca_k in [31, 128, 256]:
        add_group(pca_span_path, f"random_in_pca{pca_k}", pca_ids[:pca_k], pca_directions[:pca_k])
    return pd.DataFrame(rows), groups


def _saved_projection_bundle(
    store,
    *,
    split: str,
    probe_ids: np.ndarray,
    max_documents: int | None,
    derived_groups: list[DerivedProjectionGroup] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunk_paths = sorted(Path(store.residual_projection_split_dir(split)).glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No saved residual projection chunks found for split={split!r}.")
    requested = probe_ids.astype(str).tolist()
    derived_groups = derived_groups or []
    derived_ids = {
        str(probe_id)
        for group in derived_groups
        for probe_id in group.target_probe_ids.astype(str)
    }
    direct_ids = [probe_id for probe_id in requested if probe_id not in derived_ids]
    tokens: list[np.ndarray] = []
    context_ids: list[np.ndarray] = []
    projections: list[np.ndarray] = []
    loaded_documents = 0
    for path in chunk_paths:
        chunk = load_projection_chunk(str(path))
        index = {probe_id: idx for idx, probe_id in enumerate(chunk["probe_ids"].astype(str))}
        required = direct_ids + [
            str(probe_id)
            for group in derived_groups
            for probe_id in group.source_probe_ids.astype(str)
        ]
        missing = sorted(set(probe_id for probe_id in required if probe_id not in index))
        if missing:
            raise ValueError(f"Saved projection chunk {path} is missing selected probes: {missing}")
        take = len(chunk["tokens"])
        if max_documents is not None:
            take = min(take, max_documents - loaded_documents)
        if take <= 0:
            break
        projected_by_id = {
            probe_id: chunk["projections"][:take, :, index[probe_id]].astype(np.float32)
            for probe_id in direct_ids
        }
        for group in derived_groups:
            source = chunk["projections"][
                :take,
                :,
                [index[str(probe_id)] for probe_id in group.source_probe_ids],
            ].astype(np.float32)
            derived = np.einsum("bts,ps->btp", source, group.coefficients, optimize=True)
            projected_by_id.update(
                {
                    str(probe_id): derived[:, :, offset]
                    for offset, probe_id in enumerate(group.target_probe_ids.astype(str))
                }
            )
        tokens.append(chunk["tokens"][:take].astype(np.int64))
        context_ids.append(chunk["context_ids"][:take].astype(str))
        projections.append(np.stack([projected_by_id[probe_id] for probe_id in requested], axis=-1).astype(np.float16))
        loaded_documents += take
    if not tokens:
        raise ValueError("No saved projection documents selected.")
    return np.concatenate(tokens), np.concatenate(context_ids), np.concatenate(projections)


def _top_entries(values: np.ndarray, count: int, largest: bool) -> list[tuple[int, int, float]]:
    flat = values.reshape(-1)
    count = min(count, len(flat))
    candidate = np.argpartition(flat, -count)[-count:] if largest else np.argpartition(flat, count)[:count]
    ordered = candidate[np.argsort(flat[candidate])]
    if largest:
        ordered = ordered[::-1]
    seq_len = values.shape[1]
    return [(int(idx // seq_len), int(idx % seq_len), float(flat[idx])) for idx in ordered]


def _spans(values: np.ndarray, threshold: float, min_length: int) -> list[tuple[int, int, int, float, float]]:
    spans = []
    for doc_idx, doc_values in enumerate(values):
        mask = doc_values >= threshold
        padded = np.pad(mask.astype(np.int8), (1, 1))
        edges = np.diff(padded)
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if stop - start >= min_length:
                segment = doc_values[start:stop]
                spans.append((doc_idx, int(start), int(stop), float(segment.mean()), float(segment.sum())))
    return sorted(spans, key=lambda row: row[4], reverse=True)


def _document_level_span_metrics(
    retained: list[tuple[int, int, int, float, float]],
    *,
    n_documents: int,
    top_entries: list[tuple[int, int, float]],
    bottom_entries: list[tuple[int, int, float]],
) -> dict[str, float | int]:
    max_span_length_by_doc = np.zeros(n_documents, dtype=np.int64)
    for doc_idx, start, stop, _, _ in retained:
        max_span_length_by_doc[doc_idx] = max(max_span_length_by_doc[doc_idx], stop - start)
    covered_lengths = max_span_length_by_doc[max_span_length_by_doc > 0]

    def window_metrics(entries: list[tuple[int, int, float]], prefix: str) -> dict[str, float | int]:
        counts = pd.Series([doc_idx for doc_idx, _, _ in entries], dtype=np.int64).value_counts()
        return {
            f"n_unique_{prefix}_window_docs": int(len(counts)),
            f"fraction_unique_{prefix}_window_docs": float(len(counts) / len(entries)) if entries else 0.0,
            f"max_{prefix}_window_doc_share": float(counts.iloc[0] / len(entries)) if entries else 0.0,
        }

    return {
        "n_documents": int(n_documents),
        "n_docs_with_retained_q95_span": int(len(covered_lengths)),
        "fraction_docs_with_retained_q95_span": float(len(covered_lengths) / n_documents) if n_documents else 0.0,
        "spans_per_covered_doc": float(len(retained) / len(covered_lengths)) if len(covered_lengths) else 0.0,
        "median_doc_max_q95_span_length_covered": float(np.median(covered_lengths)) if len(covered_lengths) else 0.0,
        "q90_doc_max_q95_span_length_covered": float(np.quantile(covered_lengths, 0.90)) if len(covered_lengths) else 0.0,
        "mean_doc_max_q95_span_length_all_docs": float(max_span_length_by_doc.mean()) if n_documents else 0.0,
        **window_metrics(top_entries, "top"),
        **window_metrics(bottom_entries, "bottom"),
    }


def _decode(tokenizer, tokens: np.ndarray, start: int, stop: int) -> str:
    return tokenizer.decode(tokens[start:stop].tolist(), skip_special_tokens=True).replace("\n", " ").strip()


def _window_row(tokenizer, tokens, context_ids, probe_id, family, reason, tau, polarity, doc_idx, pos, value, radius):
    start = max(0, pos - radius)
    stop = min(tokens.shape[1], pos + radius + 1)
    return {
        "probe_id": probe_id,
        "probe_family": family,
        "selection_reason": reason,
        "tau_within": tau,
        "polarity": polarity,
        "value": value,
        "context_id": context_ids[doc_idx],
        "token_position": pos,
        "window_start": start,
        "window_end": stop,
        "decoded_window": _decode(tokenizer, tokens[doc_idx], start, stop),
    }


def _span_row(tokenizer, tokens, context_ids, probe_id, family, reason, tau, doc_idx, start, stop, mean, score):
    return {
        "probe_id": probe_id,
        "probe_family": family,
        "selection_reason": reason,
        "tau_within": tau,
        "context_id": context_ids[doc_idx],
        "span_start": start,
        "span_end": stop,
        "span_length": stop - start,
        "mean_value": mean,
        "score": score,
        "decoded_span": _decode(tokenizer, tokens[doc_idx], start, stop),
    }


def _labels(text: str) -> dict[str, float]:
    lower = text.lower()
    alnum = max(1, sum(char.isalnum() for char in text))
    words = re.findall(r"\b[a-z0-9]+\b", lower)
    counts = pd.Series(words).value_counts() if words else pd.Series(dtype=int)
    labels = {name: float(bool(re.search(pattern, lower, flags=re.I))) for name, pattern in LABEL_PATTERNS.items()}
    labels.update(
        {
            "high_digit_density": float(sum(char.isdigit() for char in text) / alnum >= 0.12),
            "high_punctuation_density": float(sum(not char.isalnum() and not char.isspace() for char in text) / max(1, len(text)) >= 0.10),
            "high_uppercase_density": float(sum(char.isupper() for char in text) / alnum >= 0.16),
            "high_repetition": float(bool(len(words) >= 20 and not counts.empty and counts.iloc[0] / len(words) >= 0.08)),
        }
    )
    return labels


def _add_label_metrics(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = pd.concat([windows.reset_index(drop=True), windows["decoded_window"].map(_labels).apply(pd.Series)], axis=1)
    label_cols = list(LABEL_PATTERNS) + [
        "high_digit_density",
        "high_punctuation_density",
        "high_uppercase_density",
        "high_repetition",
    ]
    rates = labeled.groupby(["probe_id", "polarity"])[label_cols].mean().unstack(fill_value=0.0)
    rows = []
    for probe_id, row in rates.iterrows():
        for label in label_cols:
            top = float(row.get((label, "top"), 0.0))
            bottom = float(row.get((label, "bottom"), 0.0))
            rows.append({"probe_id": probe_id, "label": label, "top_rate": top, "bottom_rate": bottom, "abs_top_bottom_gap": abs(top - bottom)})
    return labeled, pd.DataFrame(rows)


def _write_dossiers(out_dir: Path, selection: pd.DataFrame, windows: pd.DataFrame, spans: pd.DataFrame, summary: pd.DataFrame) -> None:
    dossier_dir = out_dir / "per_direction"
    dossier_dir.mkdir(parents=True, exist_ok=True)
    summary_idx = summary.set_index("probe_id")
    for _, selected in selection.iterrows():
        probe_id = selected["probe_id"]
        metric = summary_idx.loc[probe_id]
        lines = [
            f"# `{probe_id}`",
            "",
            f"family: `{selected['probe_family']}`; selected as: `{selected['selection_reason']}`; tau: `{selected['tau_within']:.1f}`",
            "",
            f"q95 spans length >= 8: `{int(metric['n_q95_spans_len_ge_8'])}`; mean retained span length: `{metric['mean_q95_span_length']:.1f}`",
            "",
            f"distinct documents with retained q95 spans: `{int(metric['n_docs_with_retained_q95_span'])}` / `{int(metric['n_documents'])}`; mean per-document maximum span length across all documents: `{metric['mean_doc_max_q95_span_length_all_docs']:.2f}`",
            "",
        ]
        for polarity in ["top", "bottom"]:
            lines.extend([f"## {polarity.title()} Token Windows", ""])
            subset = windows[(windows["probe_id"] == probe_id) & (windows["polarity"] == polarity)]
            for idx, row in enumerate(subset.itertuples(index=False), start=1):
                lines.extend([f"### {idx}. value `{row.value:.4f}`, position `{row.token_position}`", row.decoded_window, ""])
        lines.extend(["## Persistent q95 Spans", ""])
        for idx, row in enumerate(spans[spans["probe_id"] == probe_id].head(6).itertuples(index=False), start=1):
            lines.extend([f"### {idx}. len `{row.span_length}`, mean `{row.mean_value:.4f}`", row.decoded_span, ""])
        (dossier_dir / f"{probe_id}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare semantic tails of persistent, PCA, and random residual directions.")
    parser.add_argument("--config", default="configs/persistent_state/residual_geometry/pilot.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--random-count", type=int, default=5)
    parser.add_argument("--tail-windows", type=int, default=12)
    parser.add_argument("--window-radius", type=int, default=40)
    parser.add_argument("--min-span-length", type=int, default=8)
    parser.add_argument("--max-documents", type=int, default=None, help="Limit validation documents for benchmarking.")
    parser.add_argument(
        "--use-saved-projections",
        action="store_true",
        help="Read Stage B validation projection chunks instead of running model forward passes.",
    )
    parser.add_argument(
        "--include-span-samples",
        action="store_true",
        help="Also inspect locked random samples inside Q_kstar and PCA-31/128/256 spans. Requires --use-saved-projections.",
    )
    parser.add_argument("--span-sample-count", type=int, default=5)
    args = parser.parse_args()

    config, store = load_config_and_store(args.config)
    residual_cfg = require_residual_geometry_config(config)
    out_dir = Path(args.out_dir or Path(store.subspace_dir) / "semantic_axis_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = _selection_table(store, random_count=args.random_count)
    derived_groups: list[DerivedProjectionGroup] = []
    if args.include_span_samples:
        if not args.use_saved_projections:
            raise ValueError("--include-span-samples requires --use-saved-projections")
        span_selection, derived_groups = _locked_span_samples(store, count=args.span_sample_count)
        selection = pd.concat([selection, span_selection], ignore_index=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, revision=config.model.revision, local_files_only=True)

    if args.use_saved_projections:
        tokens, context_ids, projected = _saved_projection_bundle(
            store,
            split="val",
            probe_ids=selection["probe_id"].astype(str).to_numpy(),
            max_documents=args.max_documents,
            derived_groups=derived_groups,
        )
        print(f"Loaded saved projections for {len(tokens)} validation documents.", flush=True)
        data_source = "saved_stage_b_projection_chunks"
    else:
        probes = _selected_probes(store, selection)
        contexts = split_contexts(store, "val")
        if args.max_documents is not None:
            contexts = contexts.head(args.max_documents).copy()
        tokens = token_matrix(contexts)
        context_ids = contexts["context_id"].astype(str).to_numpy()
        projections = []
        from residual_geometry.residuals.provider import ResidualStreamProvider

        provider = ResidualStreamProvider(config)
        provider.load()
        batch_size = args.batch_size or residual_cfg.projections.batch_size
        started = time.monotonic()
        for start, residuals in residual_batches(provider, tokens, batch_size):
            projections.append(project_residuals(residuals, probes).astype(np.float16))
            completed = min(start + batch_size, len(tokens))
            elapsed = time.monotonic() - started
            print(f"Projected {completed}/{len(tokens)} validation documents in {elapsed:.1f}s", flush=True)
        projected = np.concatenate(projections, axis=0)
        data_source = "new_model_forward_pass"

    windows, spans, metrics = [], [], []
    selected_idx = selection.set_index("probe_id")
    for probe_idx, probe_id in enumerate(selection["probe_id"].astype(str)):
        values = projected[:, :, probe_idx].astype(np.float32)
        selected = selected_idx.loc[probe_id]
        entries_by_polarity = {
            "top": _top_entries(values, args.tail_windows, largest=True),
            "bottom": _top_entries(values, args.tail_windows, largest=False),
        }
        for polarity in ["top", "bottom"]:
            for doc_idx, pos, value in entries_by_polarity[polarity]:
                windows.append(
                    _window_row(
                        tokenizer, tokens, context_ids, probe_id, selected["probe_family"], selected["selection_reason"],
                        selected["tau_within"], polarity, doc_idx, pos, value, args.window_radius,
                    )
                )
        q95 = float(np.quantile(values, 0.95))
        retained = _spans(values, threshold=q95, min_length=args.min_span_length)
        document_metrics = _document_level_span_metrics(
            retained,
            n_documents=len(values),
            top_entries=entries_by_polarity["top"],
            bottom_entries=entries_by_polarity["bottom"],
        )
        for doc_idx, start, stop, mean, score in retained[:12]:
            spans.append(
                _span_row(
                    tokenizer, tokens, context_ids, probe_id, selected["probe_family"], selected["selection_reason"],
                    selected["tau_within"], doc_idx, start, stop, mean, score,
                )
            )
        metrics.append(
            {
                "probe_id": probe_id,
                "q05": float(np.quantile(values, 0.05)),
                "q50": float(np.quantile(values, 0.50)),
                "q95": q95,
                "projection_std": float(values.std()),
                "n_q95_spans_len_ge_8": len(retained),
                "mean_q95_span_length": float(np.mean([stop - start for _, start, stop, _, _ in retained])) if retained else 0.0,
                "top_boundary_rate_position_lt_64": float(np.mean([pos < 64 for _, pos, _ in entries_by_polarity["top"]])),
                **document_metrics,
            }
        )
    windows_df = pd.DataFrame(windows)
    spans_df = pd.DataFrame(spans)
    metrics_df = pd.DataFrame(metrics)
    labeled_df, label_gaps = _add_label_metrics(windows_df)
    max_label_gap = label_gaps.groupby("probe_id")["abs_top_bottom_gap"].max().rename("max_shallow_label_top_bottom_gap")
    summary = selection.merge(metrics_df, on="probe_id").merge(max_label_gap, on="probe_id")
    group_summary = (
        summary.groupby("selection_reason")
        .agg(
            n_directions=("probe_id", "count"),
            median_tau=("tau_within", "median"),
            median_projection_std=("projection_std", "median"),
            median_q95_span_count=("n_q95_spans_len_ge_8", "median"),
            median_q95_span_length=("mean_q95_span_length", "median"),
            median_docs_with_retained_q95_span=("n_docs_with_retained_q95_span", "median"),
            median_fraction_docs_with_retained_q95_span=("fraction_docs_with_retained_q95_span", "median"),
            median_spans_per_covered_doc=("spans_per_covered_doc", "median"),
            median_doc_max_q95_span_length_covered=("median_doc_max_q95_span_length_covered", "median"),
            median_mean_doc_max_q95_span_length_all_docs=("mean_doc_max_q95_span_length_all_docs", "median"),
            median_unique_top_window_docs=("n_unique_top_window_docs", "median"),
            median_max_top_window_doc_share=("max_top_window_doc_share", "median"),
            median_top_boundary_rate=("top_boundary_rate_position_lt_64", "median"),
            median_max_shallow_label_gap=("max_shallow_label_top_bottom_gap", "median"),
        )
        .reset_index()
    )

    selection.to_csv(out_dir / "selection.csv", index=False)
    windows_df.to_csv(out_dir / "top_bottom_token_windows.csv", index=False)
    spans_df.to_csv(out_dir / "persistent_q95_spans.csv", index=False)
    labeled_df.to_csv(out_dir / "labeled_token_windows.csv", index=False)
    label_gaps.to_csv(out_dir / "shallow_label_top_bottom_gaps.csv", index=False)
    summary.to_csv(out_dir / "direction_summary.csv", index=False)
    group_summary.to_csv(out_dir / "group_summary.csv", index=False)
    summary[
        [
            "probe_id",
            "probe_family",
            "selection_reason",
            "tau_within",
            "n_documents",
            "n_docs_with_retained_q95_span",
            "fraction_docs_with_retained_q95_span",
            "spans_per_covered_doc",
            "median_doc_max_q95_span_length_covered",
            "q90_doc_max_q95_span_length_covered",
            "mean_doc_max_q95_span_length_all_docs",
            "n_unique_top_window_docs",
            "fraction_unique_top_window_docs",
            "max_top_window_doc_share",
            "n_unique_bottom_window_docs",
            "fraction_unique_bottom_window_docs",
            "max_bottom_window_doc_share",
        ]
    ].to_csv(out_dir / "document_dedup_direction_summary.csv", index=False)
    group_summary[
        [
            "selection_reason",
            "n_directions",
            "median_tau",
            "median_docs_with_retained_q95_span",
            "median_fraction_docs_with_retained_q95_span",
            "median_spans_per_covered_doc",
            "median_doc_max_q95_span_length_covered",
            "median_mean_doc_max_q95_span_length_all_docs",
            "median_unique_top_window_docs",
            "median_max_top_window_doc_share",
        ]
    ].to_csv(out_dir / "document_dedup_group_summary.csv", index=False)
    _write_dossiers(out_dir, selection, windows_df, spans_df, summary)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "validation_documents": len(tokens),
                "directions": len(selection),
                "projection_data_source": data_source,
                "included_locked_span_samples": bool(args.include_span_samples),
                "span_sample_count_per_family": int(args.span_sample_count) if args.include_span_samples else 0,
                "selection_reason_counts": selection["selection_reason"].value_counts().to_dict(),
                "notes": [
                    "PCA axes include the union of top-10 explained-variance axes and top-10 validation-lifetime PCA axes.",
                    "Persistent references are the top-10 time-lagged axes by validation lifetime.",
                    "Random controls are a fixed seeded sample, not selected for apparent semantics or lifetime.",
                    "All text-label metrics are exploratory shallow regex/density checks.",
                    "Document-level q95-span metrics collapse retained spans to distinct documents before comparison.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(group_summary.to_string(index=False))
    print(f"\nWrote comparison artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
