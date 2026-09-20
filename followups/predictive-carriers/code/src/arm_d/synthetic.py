"""Synthetic worlds for Arm D qualification (R1 §31).

Every world produces a :class:`~arm_d.pipeline.PairedDataset` in aperture
coordinates, so the qualification harness exercises the *production estimator*
rather than a simplified stand-in.

The worlds that must FAIL matter as much as the worlds that must pass.  A
pipeline that cannot produce a null is not an instrument, and the paired design
changes what "must fail" means: because the update sequence is identical in the
two runs, a local-window confound, a position drift, or a document-held topic
cancels in the difference and must leave no incremental structure.

``S-DT`` is a twelfth, qualification-only world: it plants genuine
source-to-destination transport between two non-overlapping planes and is used
solely to calibrate the shared-space MSC gate that R1 §31.1 requires
qualification to freeze.  It carries no power or false-positive bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm_d.pipeline import PairedDataset

WORLDS = (
    "S-D1",
    "S-D2",
    "S-D3",
    "S-D4",
    "S-D5",
    "S-D6",
    "S-D7",
    "S-D8",
    "S-D9",
    "S-D10",
    "S-D11",
)
CALIBRATION_WORLDS = ("S-DT",)

MUST_DETECT = ("S-D1", "S-D2", "S-D3", "S-D8", "S-D11")

#: Worlds where D1 itself must stay null.  R1 §31 puts only the pure
#: local-window confound and the pure position drift here.  ``S-D6`` (a
#: document-held topic) is NOT a D1 false positive: R1 requires observational
#: prediction to be possible there and only the causal-relay label to be
#: absent, so it belongs with the causal-label worlds below.
MUST_NOT_DETECT = ("S-D4", "S-D5")

#: Worlds whose bar is on a causal or compositional label, not on D1.
CAUSAL_LABEL_WORLDS = ("S-D6", "S-D9", "S-D10")

#: Worlds where the nested family must show coordinate mixing (R1 §11.2).
#: ``S-D1`` and ``S-D3`` are scalar and must NOT show it.
MUST_SHOW_MIXING = ("S-D2", "S-D8", "S-D11")
MUST_NOT_SHOW_MIXING = ("S-D1", "S-D3")

CALIBRATE_THRESHOLD_ON = "S-D4"


@dataclass(frozen=True)
class WorldConfig:
    n_documents: int = 240
    width: int = 64
    rank: int = 4
    lags: tuple[int, ...] = (1, 2, 4, 8)
    history_offsets: tuple[int, ...] = (-4, -3, -2, -1, 0)
    noise: float = 0.35
    #: Power-law falloff of the ambient (non-planted) variance across aperture
    #: coordinates.  Zero gives isotropic noise, which penalizes a wider
    #: aperture purely by adding energy -- an artifact of the generator, not a
    #: property of residual coordinates.  Real aperture coordinates are PCA
    #: coordinates with a decaying spectrum, so the aperture-qualification
    #: world uses ``spectral_decay=1.0``.
    spectral_decay: float = 0.0
    rho: float = 0.92
    omega: float = 0.35
    suffix_length: int = 64


def _plant(width: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(width, rank)))[0]


def _splits(n: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.array(["train"] * n, dtype=object)
    order = rng.permutation(n)
    labels[order[int(0.6 * n) : int(0.8 * n)]] = "val"
    labels[order[int(0.8 * n) :]] = "test"
    return labels.astype(str)


def _rotation(omega: float) -> np.ndarray:
    return np.array([[np.cos(omega), -np.sin(omega)], [np.sin(omega), np.cos(omega)]])


def _block_operator(config: WorldConfig, kind: str, rng: np.random.Generator) -> np.ndarray:
    r = config.rank
    if kind == "scalar":
        return config.rho * np.eye(r)
    if kind == "sign":
        return -config.rho * np.eye(r)
    if kind == "rotation":
        operator = np.zeros((r, r))
        for start in range(0, r - 1, 2):
            operator[start : start + 2, start : start + 2] = config.rho * _rotation(config.omega)
        if r % 2:
            operator[-1, -1] = config.rho
        return operator
    if kind == "degenerate":
        operator = np.zeros((r, r))
        for index, start in enumerate(range(0, r - 1, 2)):
            omega = config.omega * (1.0 + 0.02 * index)
            operator[start : start + 2, start : start + 2] = config.rho * _rotation(omega)
        if r % 2:
            operator[-1, -1] = config.rho
        return operator
    if kind == "mixing":
        base = rng.normal(size=(r, r)) / np.sqrt(r)
        values = np.linalg.eigvals(base)
        return config.rho * base / max(float(np.max(np.abs(values))), 1e-9)
    raise ValueError(f"unknown operator kind: {kind!r}")


def _dataset(
    source: np.ndarray,
    history: np.ndarray,
    future: np.ndarray,
    config: WorldConfig,
    rng: np.random.Generator,
) -> PairedDataset:
    n = source.shape[0]
    document_ids = np.array([f"doc{index:05d}" for index in range(n)])
    return PairedDataset(
        document_ids=document_ids,
        condition=np.array(["main"] * n),
        suffix_length=np.full(n, config.suffix_length, dtype=np.int64),
        band=np.arange(n) % 8,
        split=_splits(n, rng),
        source=source,
        history=history,
        future=future,
        lags=config.lags,
        history_offsets=np.asarray(config.history_offsets, dtype=np.int64),
    )


def _trajectory(
    operator: np.ndarray,
    config: WorldConfig,
    rng: np.random.Generator,
    *,
    non_markov: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latent trajectory over history offsets and evaluated lags.

    ``non_markov`` adds a dependence on an older state through a *different*
    operator.  Adding the same operator at an older lag would leave the process
    approximately geometric and would not break the semigroup, which is the
    property S-D7 has to violate.
    """

    n, r = config.n_documents, config.rank
    depth = int(-min(config.history_offsets))
    horizon = int(max(config.lags))
    total = depth + horizon + 1
    secondary = rng.normal(size=(r, r)) / np.sqrt(r)
    secondary = 0.85 * secondary / max(float(np.max(np.abs(np.linalg.eigvals(secondary)))), 1e-9)
    states = np.zeros((n, total, r))
    states[:, 0] = rng.normal(size=(n, r))
    for step in range(1, total):
        drive = states[:, step - 1] @ operator.T
        if non_markov and step >= 3:
            drive = 0.5 * drive + 0.9 * (states[:, step - 3] @ secondary.T)
        states[:, step] = drive + config.noise * rng.normal(size=(n, r))
    history_index = np.array([depth + offset for offset in config.history_offsets])
    future_index = np.array([depth + lag for lag in config.lags])
    return states[:, history_index], states[:, depth], states[:, future_index]


def world_plant(world: str, config: WorldConfig, seed: int) -> np.ndarray:
    """The planted source subspace for a world, for containment diagnostics.

    Regenerated from the same seed and the same first draw as
    :func:`build_world`, so it is the exact subspace that world planted.
    """

    return _plant(config.width, config.rank, np.random.default_rng(seed))


def build_world(world: str, config: WorldConfig, seed: int) -> PairedDataset:
    """Generate one replicate of a named world in aperture coordinates."""

    rng = np.random.default_rng(seed)
    n, p, r = config.n_documents, config.width, config.rank
    plant = _plant(p, r, rng)
    profile = (np.arange(1, p + 1, dtype=np.float64)) ** (-0.5 * config.spectral_decay)
    profile = profile / profile[0]
    ambient = config.noise * rng.normal(size=(n, len(config.lags), p)) * profile
    ambient_history = config.noise * rng.normal(size=(n, len(config.history_offsets), p)) * profile

    def embed(history_latent, source_latent, future_latent, destination_plant=None):
        destination = plant if destination_plant is None else destination_plant
        # The history is the SOURCE-hook trajectory and its last slot is the
        # source state itself, so persistence is the operator with A = I.
        history = np.einsum("nhr,pr->nhp", history_latent, plant) + ambient_history
        history[:, -1, :] = source_latent @ plant.T + ambient_history[:, -1, :]
        source = history[:, -1, :]
        future = np.einsum("nkr,pr->nkp", future_latent, destination) + ambient
        return _dataset(source, history, future, config, rng)

    if world in {"S-D1", "S-D2", "S-D3", "S-D8", "S-D11"}:
        kind = {
            "S-D1": "scalar",
            "S-D2": "rotation",
            "S-D3": "sign",
            "S-D8": "mixing",
            "S-D11": "degenerate",
        }[world]
        operator = _block_operator(config, kind, rng)
        history, source, future = _trajectory(operator, config, rng)
        return embed(history, source, future)

    if world == "S-D7":
        operator = _block_operator(config, "mixing", rng)
        history, source, future = _trajectory(operator, config, rng, non_markov=True)
        return embed(history, source, future)

    if world == "S-D4":
        # Local-window confound.  The confound is identical in the two paired
        # runs, so it cancels in the difference: what remains is independent
        # noise at source and destination.
        history = rng.normal(size=(n, len(config.history_offsets), p)) * config.noise
        source = history[:, -1, :]
        future = rng.normal(size=(n, len(config.lags), p)) * config.noise
        return _dataset(source, history, future, config, rng)

    if world == "S-D5":
        # Pure position drift: a smooth position-dependent offset shared by
        # source, history and future.  Persistence explains all of it, so the
        # incremental statistic must be null.
        drift = rng.normal(size=(1, p)) * np.linspace(1.0, 1.4, n)[:, None]
        history = drift[:, None, :] + config.noise * rng.normal(size=(n, len(config.history_offsets), p))
        source = history[:, -1, :]
        future = drift[:, None, :] + config.noise * rng.normal(size=(n, len(config.lags), p))
        return _dataset(source, history, future, config, rng)

    if world == "S-D6":
        # Document-held topic: a per-document constant latent with no
        # propagated token-to-token state.
        latent = rng.normal(size=(n, r)) @ plant.T
        history = latent[:, None, :] + config.noise * rng.normal(size=(n, len(config.history_offsets), p))
        source = history[:, -1, :]
        future = latent[:, None, :] + config.noise * rng.normal(size=(n, len(config.lags), p))
        return _dataset(source, history, future, config, rng)

    if world in {"S-D9", "S-D10"}:
        # Observationally identical; they differ only under intervention, which
        # :func:`causal_response` supplies.
        operator = _block_operator(config, "mixing", rng)
        history, source, future = _trajectory(operator, config, rng)
        return embed(history, source, future)

    if world == "S-DT":
        # Genuine source-to-destination transport between non-overlapping
        # planes: used only to calibrate the shared-space MSC gate.
        destination_plant = _plant(p, r, rng)
        destination_plant = np.linalg.qr(
            destination_plant - plant @ (plant.T @ destination_plant)
        )[0]
        operator = _block_operator(config, "mixing", rng)
        history, source, future = _trajectory(operator, config, rng)
        return embed(history, source, future, destination_plant=destination_plant)

    raise ValueError(f"unknown world: {world!r}")


def world_operator(world: str, config: WorldConfig, seed: int) -> np.ndarray:
    """The planted latent operator, for causal-world simulation."""

    rng = np.random.default_rng(seed)
    _plant(config.width, config.rank, rng)
    kind = {
        "S-D1": "scalar",
        "S-D2": "rotation",
        "S-D3": "sign",
        "S-D11": "degenerate",
    }.get(world, "mixing")
    return _block_operator(config, kind, rng)


def causal_response(
    world: str,
    operator: np.ndarray,
    perturbation: np.ndarray,
    lags: tuple[int, ...],
    rng: np.random.Generator,
    *,
    noise: float = 0.05,
) -> np.ndarray:
    """Simulated destination shift for the causal qualification worlds.

    ``S-D9`` relays the perturbation according to the planted operator.
    ``S-D10`` reconstructs each future state from broader history instead, so
    the perturbation is erased: observationally the two worlds are identical,
    and only the intervention separates them.
    """

    responses = []
    for lag in lags:
        propagated = perturbation @ np.linalg.matrix_power(operator, lag).T
        if world == "S-D10":
            propagated = propagated * 0.02
        responses.append(propagated + noise * rng.normal(size=propagated.shape))
    return np.stack(responses, axis=1)
