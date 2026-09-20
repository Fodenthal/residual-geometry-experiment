"""Pure helpers for R1.9 content-vs-document-fingerprint identification."""
from __future__ import annotations

import re
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd


SOURCE_RULES = (
    ("government", (".gov", "government", "gov.")),
    ("education", (".edu", ".ac.", "university", "college", "school")),
    ("forum_or_community", ("forum", "board", "reddit", "stackexchange", "discussion", "community")),
    ("blog_or_personal", ("blog", "wordpress", "blogspot", "tumblr", "livejournal", "medium.com")),
    ("news_media", ("news", "times", "journal", "tribune", "daily", "post.com")),
    ("reference_or_docs", ("wiki", "docs.", "documentation", "reference", "encyclopedia")),
    ("commerce", ("shop", "store", "amazon", "product", "commerce", "market")),
)


def source_type(host: str) -> str:
    """Frozen, deterministic coarse source type derived only from URL host metadata."""
    value = str(host).lower().strip()
    for label, needles in SOURCE_RULES:
        if any(needle in value for needle in needles):
            return label
    if value.endswith(".org"):
        return "organization"
    return "other_web"


def suffix_class(host: str) -> str:
    parts = str(host).lower().strip(".").split(".")
    suffix = parts[-1] if parts else "unknown"
    if suffix in {"com", "org", "net", "edu", "gov", "io", "info"}:
        return suffix
    if len(suffix) == 2:
        return "country_code"
    return "other"


def structural_features(text: str, url: str) -> dict[str, float]:
    """Low-capacity style/template summaries; no text embeddings or lexical content."""
    value = str(text)
    n = max(len(value), 1)
    lines = value.splitlines() or [value]
    words = re.findall(r"\b\w+\b", value)
    word_count = max(len(words), 1)
    sentences = [x for x in re.split(r"[.!?]+", value) if x.strip()]
    sentence_lengths = np.asarray([len(re.findall(r"\b\w+\b", x)) for x in sentences], dtype=float)
    parsed = urlparse(str(url))
    list_lines = sum(bool(re.match(r"^\s*(?:[-*+] |\d+[.)]\s)", line)) for line in lines)
    heading_lines = sum(bool(line.strip()) and (line.strip().isupper() or len(line.strip().split()) <= 6 and line.rstrip().endswith(":")) for line in lines)
    return {
        "char_count_log": float(np.log1p(len(value))),
        "word_count_log": float(np.log1p(len(words))),
        "line_count_log": float(np.log1p(len(lines))),
        "newline_density": value.count("\n") / n,
        "mean_line_length": float(np.mean([len(x) for x in lines])),
        "list_line_fraction": list_lines / max(len(lines), 1),
        "heading_line_fraction": heading_lines / max(len(lines), 1),
        "punctuation_fraction": sum(not c.isalnum() and not c.isspace() for c in value) / n,
        "digit_fraction": sum(c.isdigit() for c in value) / n,
        "uppercase_fraction": sum(c.isupper() for c in value) / n,
        "question_fraction": value.count("?") / n,
        "exclamation_fraction": value.count("!") / n,
        "quote_fraction": sum(value.count(c) for c in ('"', "'", "“", "”")) / n,
        "colon_semicolon_fraction": (value.count(":") + value.count(";")) / n,
        "mean_sentence_words": float(sentence_lengths.mean()) if len(sentence_lengths) else float(word_count),
        "sd_sentence_words": float(sentence_lengths.std()) if len(sentence_lengths) else 0.0,
        "numeric_token_fraction": sum(w.isdigit() for w in words) / word_count,
        "url_length_log": float(np.log1p(len(str(url)))),
        "url_path_depth": float(len([x for x in parsed.path.split("/") if x])),
        "url_has_query": float(bool(parsed.query)),
        "url_https": float(parsed.scheme.lower() == "https"),
    }


def collapse_from_fit(values: np.ndarray, fit_indices: np.ndarray, minimum: int = 20) -> tuple[np.ndarray, list[str]]:
    raw = np.asarray(values, dtype=str)
    counts = Counter(raw[np.asarray(fit_indices, dtype=int)])
    supported = sorted(k for k, v in counts.items() if v >= minimum)
    collapsed = np.asarray([x if x in supported else "__LOW_SUPPORT__" for x in raw], dtype=object)
    categories = supported + (["__LOW_SUPPORT__"] if np.any(collapsed == "__LOW_SUPPORT__") else [])
    return collapsed.astype(str), categories


def one_hot_frozen(values: np.ndarray, categories: list[str]) -> np.ndarray:
    raw = np.asarray(values, dtype=str)
    return np.column_stack([raw == category for category in categories]).astype(np.float64)


def fingerprint_frame(frame: pd.DataFrame, fit_indices: np.ndarray) -> tuple[pd.DataFrame, dict]:
    required = {"document_id", "url", "host", "text", "register", "formatting"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing fingerprint columns: {missing}")
    source = np.asarray([source_type(x) for x in frame.host], dtype=str)
    suffix = np.asarray([suffix_class(x) for x in frame.host], dtype=str)
    register, register_categories = collapse_from_fit(frame.register.astype(str).to_numpy(), fit_indices)
    formatting, formatting_categories = collapse_from_fit(frame.formatting.astype(str).to_numpy(), fit_indices)
    source, source_categories = collapse_from_fit(source, fit_indices)
    suffix, suffix_categories = collapse_from_fit(suffix, fit_indices)
    numeric = pd.DataFrame([structural_features(t, u) for t, u in zip(frame.text, frame.url)])
    result = pd.DataFrame({
        "document_id": frame.document_id.astype(str), "host": frame.host.astype(str),
        "source_type": source, "host_suffix_class": suffix,
        "register": register, "formatting": formatting,
    })
    result = pd.concat([result, numeric], axis=1)
    schema = {
        "families": ["source/domain", "genre/register", "format/template"],
        "source_observation": "explicit URL and host plus frozen coarse metadata-derived source type; no activation-derived source features",
        "source_identity_limitation": "exact hosts are retained for provenance but not one-hot encoded because nearly all hosts are singletons",
        "categorical": {
            "source_type": source_categories, "host_suffix_class": suffix_categories,
            "register": register_categories, "formatting": formatting_categories,
        },
        "numeric": list(numeric.columns), "minimum_fit_support": 20,
    }
    return result, schema


def final_outcome(adequate: bool, survives: bool, geometry: bool, both_in_random: bool, fingerprint_improvement: float) -> tuple[str, str]:
    if not adequate:
        return "FINGERPRINT_CONTROL_INADEQUATE", "repair only the missing fingerprint measurement; do not interpret content survival"
    if survives and geometry:
        return "CONTENT_STATE_IDENTIFIED", "design one within-document changing semantic-state target"
    if both_in_random and fingerprint_improvement > 0:
        return "DOCUMENT_FINGERPRINT_DOMINANT", "reinterpret the slow region as document fingerprint/register/source state"
    return "TOPIC_PARTLY_FINGERPRINT_MEDIATED", "stop; do not broaden labels automatically"
