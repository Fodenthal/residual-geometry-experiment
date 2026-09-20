"""Known-answer worlds for the estimators, including amendment A6's anisotropic null.

Q0 to Q3 are estimation tasks, so what they need is not a power campaign but proof that the
implementation recovers a planted answer and refuses an absent one.  Every world here is
generated in the aperture's own shape and then run through the REAL estimator and the REAL
scoring path -- the same ``fit_frame``, ``carrier_family`` and ``score_carrier`` the
confirmatory run uses.

The world that matters most is Q-S6.  The source suite's only no-prediction world is
isotropic, and Q-W is designed to look where destination variance is small; on a steeply
decaying spectrum the small-variance tail is exactly where the covariance estimate is worst.
Q-S0 cannot detect a whitened estimator that manufactures a carrier out of that tail.  Q-S6
is that world: the real aperture's spectrum, and no predictive structure anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from arm_d_r24.contract import LAGS_Q, SEED
from arm_d_r24.data import Pool, synthetic_pool

WORLD_NAMES = ("Q_S0", "Q_S1", "Q_S2", "Q_S3", "Q_S4", "Q_S5", "Q_S6")


@dataclass
class World:
    name: str
    description: str
    pool: Pool
    truth: dict[str, object] = field(default_factory=dict)
    expectation: str = ""


def _spectrum(kind: str, width: int) -> np.ndarray:
    if kind == "isotropic":
        return np.ones(width)
    if kind == "aperture":
        # The real aperture's shape: a steep head holding about two thirds of the variance in
        # the leading sixteenth of the directions (R2.3c measured 0.6722 for the leading 16 of
        # 256), and a flat tail holding the rest.  A pure exponential would put the tail
        # directions 1e-7 below the head, where nothing is estimable and the "low-variance
        # predictive carrier" world would test the ridge rather than the estimator.
        head = max(int(round(width / 16)), 2)
        values = np.zeros(width, dtype=np.float64)
        decay = 8.0 / head
        front = np.exp(-decay * np.arange(head))
        values[:head] = 0.672 * front / front.sum()
        values[head:] = 0.328 / float(width - head)
        return values * width
    raise ValueError(f"unknown spectrum {kind!r}")


def _draw(rng: np.random.Generator, n: int, spectrum: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return (rng.normal(size=(n, spectrum.shape[0])) * np.sqrt(spectrum)) @ rotation.T


def build_world(
    name: str,
    *,
    n_documents: int = 2000,
    width: int = 256,
    lags: Sequence[int] = LAGS_Q,
    seed: int = SEED,
    planted_rank: int = 4,
    persistence_tau: float = 24.0,
    noise_scale: float = 1.0,
    signal_scale: float = 2.5,
) -> World:
    """Simulate one world.  ``truth`` records the planted objects the checks compare against."""

    rng = np.random.default_rng(seed + abs(hash(name)) % 100000)
    lags = tuple(int(value) for value in lags)
    rotation = np.linalg.qr(rng.normal(size=(width, width)))[0]

    spectrum_kind = "isotropic" if name in {"Q_S0", "Q_S1", "Q_S4"} else "aperture"
    spectrum = _spectrum(spectrum_kind, width)
    x = _draw(rng, n_documents, spectrum, rotation)
    previous = 0.7 * x + np.sqrt(1 - 0.7**2) * _draw(rng, n_documents, spectrum, rotation)

    # Planted source functionals.  The estimator recovers span(U) as a set of linear
    # functionals of x, so the truth object is U itself in aperture coordinates.
    def eigen_directions(indices: Sequence[int]) -> np.ndarray:
        return np.ascontiguousarray(rotation[:, list(indices)])

    truth: dict[str, object] = {"spectrum_kind": spectrum_kind, "planted_rank": int(planted_rank)}
    carriers: list[tuple[np.ndarray, np.ndarray, float]] = []  # (U, destination map, strength)
    # Block positions are fractions of the aperture so a small world keeps the same shape.
    mid_start = int(round(0.23 * width))
    second_start = int(round(0.39 * width))
    tail_start = int(round(0.94 * width)) - planted_rank
    high = eigen_directions(range(planted_rank))                          # top-variance block
    middle = eigen_directions(range(mid_start, mid_start + planted_rank))  # mid-variance block
    tail = eigen_directions(range(tail_start, tail_start + planted_rank))  # low-variance tail

    destination_high = eigen_directions(range(planted_rank, 2 * planted_rank))
    destination_tail = eigen_directions(
        range(tail_start - 2 * planted_rank, tail_start - planted_rank)
    )

    if name == "Q_S0":
        truth["planted_carrier"] = None
        expectation = "no stable carrier: the split-half overlap must not clear the null"
    elif name == "Q_S1":
        carriers.append((middle, destination_high, 0.6))
        truth["planted_carrier"] = middle
        expectation = "both estimators recover the planted span under isotropic covariance"
    elif name == "Q_S2":
        carriers.append((middle, destination_tail, 0.6))
        truth["planted_carrier"] = middle
        truth["nuisance_high_variance_block"] = high
        expectation = (
            "the high-variance block carries no predictive relation; Q-W must recover the "
            "planted middle block rather than the high-variance one"
        )
    elif name == "Q_S3":
        carriers.append((tail, destination_tail, 0.75))
        truth["planted_carrier"] = tail
        expectation = "Q-W must recover the low-variance strongly predictive carrier"
    elif name == "Q_S4":
        # The two realizations predict DIFFERENT future latents at similar strength.  Pointing
        # both at one latent would make their union a single degenerate predictive space, and
        # a rank-r estimator would return an arbitrary mixture rather than one of them.
        second = eigen_directions(range(second_start, second_start + planted_rank))
        carriers.append((middle, destination_high, 0.60))
        carriers.append((second, destination_tail, 0.50))
        truth["planted_carrier"] = middle
        truth["planted_second_carrier"] = second
        expectation = "complement refit must recover the second redundant carrier"
    elif name == "Q_S5":
        carriers.append((high, destination_high, 0.25))     # big raw energy, weak normalized
        carriers.append((tail, destination_tail, 0.75))     # small raw energy, strong normalized
        truth["planted_energy_carrier"] = high
        truth["planted_whitened_carrier"] = tail
        expectation = "the pipeline reports metric dependence rather than a consensus basis"
    elif name == "Q_S6":
        truth["planted_carrier"] = None
        expectation = (
            "amendment A6: a steeply decaying spectrum with no predictive structure; Q-W "
            "must not report a stable carrier out of the poorly estimated tail"
        )
    else:
        raise ValueError(f"unknown world {name!r}")

    offsets: dict[int, np.ndarray] = {0: x, -1: previous}
    for lag in lags:
        rho = float(np.exp(-lag / persistence_tau))
        signal = np.zeros_like(x)
        for basis, destination, strength in carriers:
            latent = x @ basis
            decay = float(np.exp(-lag / (persistence_tau * 1.5)))
            signal += float(signal_scale) * strength * decay * (latent @ destination.T)
        noise = noise_scale * _draw(rng, n_documents, spectrum, rotation)
        offsets[int(lag)] = rho * x + signal + noise

    truth["carrier_count"] = len(carriers)
    truth["signal_scale"] = float(signal_scale)
    return World(
        name=name,
        description=expectation,
        pool=synthetic_pool(offsets, name=f"world_{name}", split_salt=f"arm_d_r2_4_{name}"),
        truth=truth,
        expectation=expectation,
    )


def recovery_check(
    recovered: np.ndarray, planted: np.ndarray | None, *, width: int
) -> dict[str, object]:
    """MSC of the recovered carrier against the planted span, with the Haar reference."""

    from arm_d_r24.geometry import mean_subspace_correlation, principal_angles

    if planted is None:
        return {"planted": None}
    planted = np.asarray(planted, dtype=np.float64)
    rank = min(recovered.shape[1], planted.shape[1])
    left = recovered[:, :rank]
    right = planted[:, :rank]
    angles = principal_angles(left, right)
    return {
        "rank_compared": int(rank),
        "mean_subspace_correlation": mean_subspace_correlation(left, right),
        "analytic_haar_expectation": float(rank / width),
        "max_principal_angle_degrees": float(angles.max()),
    }
