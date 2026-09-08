from __future__ import annotations

import numpy as np

from residual_geometry.residuals.probes import (
    concatenate_probe_sets,
    load_probe_set,
    probe_set_from_directions,
    random_residual_probe_set,
    save_probe_set,
)
from residual_geometry.residuals.projections import project_residuals, save_projection_chunk, load_projection_chunk


def test_random_residual_probes_are_unit_norm_and_deterministic() -> None:
    a = random_residual_probe_set(d_model=8, count=5, seed=3)
    b = random_residual_probe_set(d_model=8, count=5, seed=3)
    np.testing.assert_allclose(np.linalg.norm(a.directions, axis=1), np.ones(5), atol=1e-6)
    np.testing.assert_allclose(a.directions, b.directions)
    assert a.probe_family.tolist() == ["random"] * 5


def test_project_residuals_shape_and_values() -> None:
    probes = probe_set_from_directions(np.eye(3, dtype=np.float32)[:2], family="test", id_prefix="p")
    residuals = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    projected = project_residuals(residuals, probes)
    assert projected.shape == (2, 4, 2)
    np.testing.assert_allclose(projected[..., 0], residuals[..., 0])
    np.testing.assert_allclose(projected[..., 1], residuals[..., 1])


def test_concatenate_probe_sets_preserves_metadata() -> None:
    a = random_residual_probe_set(d_model=4, count=2, seed=0)
    b = probe_set_from_directions(np.ones((1, 4), dtype=np.float32), family="ones", id_prefix="one")
    out = concatenate_probe_sets([a, b])
    assert out.directions.shape == (3, 4)
    assert out.probe_family.tolist() == ["random", "random", "ones"]


def test_probe_set_roundtrip_preserves_long_family_names(tmp_path) -> None:
    probes = probe_set_from_directions(
        np.eye(4, dtype=np.float32)[:2],
        family="random_heldout",
        id_prefix="heldout_random_residual",
    )
    path = tmp_path / "probes.npz"
    save_probe_set(probes, str(path))

    loaded = load_probe_set(str(path))

    assert loaded.probe_family.tolist() == ["random_heldout", "random_heldout"]


def test_projection_chunk_preserves_token_order(tmp_path) -> None:
    probes = probe_set_from_directions(np.eye(2, dtype=np.float32), family="eye", id_prefix="eye")
    tokens = np.array([[10, 11, 12, 13]], dtype=np.int64)
    projections = np.arange(1 * 4 * 2, dtype=np.float32).reshape(1, 4, 2)
    path = tmp_path / "chunk.npz"
    save_projection_chunk(
        str(path),
        context_ids=np.array(["ctx"]),
        tokens=tokens,
        probes=probes,
        projections=projections,
        storage_dtype="float32",
    )
    loaded = load_projection_chunk(str(path))
    np.testing.assert_array_equal(loaded["tokens"], tokens)
    np.testing.assert_allclose(loaded["projections"], projections)
