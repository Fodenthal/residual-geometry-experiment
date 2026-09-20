import json
import os

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    """Create directory and all parents if they do not exist."""
    os.makedirs(path, exist_ok=True)


def save_npy(array: np.ndarray, path: str) -> None:
    """Save a numpy array to disk, creating parent directories as needed."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    np.save(path, array)


def load_npy(path: str) -> np.ndarray:
    """Load a numpy array from disk."""
    return np.load(path, allow_pickle=False)


def save_json(data: dict, path: str) -> None:
    """Save a dict as JSON to disk, creating parent directories as needed."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    """Load a JSON file from disk and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(rows: list[dict], path: str) -> None:
    """Save a sequence of dict rows as JSONL."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame as Parquet to disk, creating parent directories as needed."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    """Load a Parquet file from disk and return as DataFrame."""
    return pd.read_parquet(path)
