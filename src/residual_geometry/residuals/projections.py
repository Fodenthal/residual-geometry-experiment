from __future__ import annotations

import numpy as np

from residual_geometry.residuals.probes import ResidualProbeSet


def project_residuals(residuals: np.ndarray, probes: ResidualProbeSet) -> np.ndarray:
    """Project residuals with shape (docs, positions, d_model) onto probe directions."""
    if residuals.ndim != 3:
        raise ValueError("residuals must have shape (docs, positions, d_model)")
    if residuals.shape[-1] != probes.d_model:
        raise ValueError(f"residual d_model={residuals.shape[-1]} does not match probes d_model={probes.d_model}")
    return np.einsum("btd,pd->btp", residuals.astype(np.float32, copy=False), probes.directions, optimize=True)


def save_projection_chunk(
    path: str,
    context_ids: np.ndarray,
    tokens: np.ndarray,
    probes: ResidualProbeSet,
    projections: np.ndarray,
    storage_dtype: str = "float16",
    **metadata: object,
) -> None:
    import os

    from residual_geometry.utils.io import ensure_dir

    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    dtype = np.float16 if storage_dtype == "float16" else np.float32
    np.savez_compressed(
        path,
        context_ids=context_ids.astype(str),
        tokens=tokens.astype(np.int64),
        probe_ids=probes.probe_ids.astype(str),
        probe_family=probes.probe_family.astype(str),
        projections=projections.astype(dtype),
        **metadata,
    )


def load_projection_chunk(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
