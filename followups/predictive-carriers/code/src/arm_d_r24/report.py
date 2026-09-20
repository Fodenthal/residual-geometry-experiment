"""Claim resolution for the R2.4 report contract.

Every number the page renders comes from a run artifact through a named claim, so the page
cannot drift from the run tree: the builder resolves each claim by reading the artifact, the
renderer substitutes only resolved claims, and the verifier re-resolves every claim from the
same artifacts and exits non-zero on any disagreement.  A claim is either

``source``   a pointer into one artifact, or
``derived``  a named formula over other claims, recomputed at verification time.

The pointer syntax is a slash-joined path of dictionary keys and list indices, so a claim
records exactly where its value lives rather than describing it in prose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "Claim",
    "ClaimError",
    "DERIVATIONS",
    "format_value",
    "resolve_claims",
    "resolve_pointer",
]


class ClaimError(RuntimeError):
    """A claim could not be resolved, which is a build failure rather than a warning."""


@dataclass(frozen=True)
class Claim:
    """One renderable fact.

    ``key``      the identifier the page uses.
    ``file``     artifact path relative to the run tree, or ``None`` for a derived claim.
    ``pointer``  slash-joined key path inside that artifact.
    ``kind``     ``"number"``, ``"integer"``, ``"label"``, ``"text"`` or ``"boolean"``.
    ``digits``   significant digits for a rendered number; four unless stated.
    ``note``     what the fact means, in one line, for the manifest reader.
    ``formula``  derivation name for a derived claim.
    ``inputs``   claim keys the derivation consumes.
    """

    key: str
    kind: str = "number"
    file: str | None = None
    pointer: str = ""
    digits: int = 4
    note: str = ""
    formula: str | None = None
    inputs: Sequence[str] = field(default_factory=tuple)


def resolve_pointer(document: Any, pointer: str) -> Any:
    """Walk ``pointer`` through nested mappings and sequences."""

    current = document
    if not pointer:
        return current
    for step in pointer.split("/"):
        if isinstance(current, Mapping):
            if step not in current:
                raise ClaimError(f"key {step!r} absent; available: {sorted(current)[:12]}")
            current = current[step]
        elif isinstance(current, Sequence) and not isinstance(current, str):
            try:
                current = current[int(step)]
            except (ValueError, IndexError) as error:
                raise ClaimError(f"index {step!r} invalid for a sequence of {len(current)}") from error
        else:
            raise ClaimError(f"cannot descend into {type(current).__name__} at {step!r}")
    return current


def _coerce(value: Any, kind: str, key: str) -> Any:
    if kind in ("number",):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ClaimError(f"claim {key} is {value!r}, which is not a number")
        return float(value)
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and float(value).is_integer():
                return int(value)
            raise ClaimError(f"claim {key} is {value!r}, which is not an integer")
        return int(value)
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ClaimError(f"claim {key} is {value!r}, which is not a boolean")
        return bool(value)
    if kind in ("label", "text"):
        if not isinstance(value, str):
            raise ClaimError(f"claim {key} is {value!r}, which is not text")
        return value
    raise ClaimError(f"claim {key} has unknown kind {kind!r}")


#: Derivations are named rather than inlined so the verifier recomputes the same arithmetic
#: instead of trusting a stored result.
DERIVATIONS: dict[str, Callable[..., float]] = {
    "ratio": lambda numerator, denominator: float(numerator) / float(denominator),
    "difference": lambda left, right: float(left) - float(right),
    "product": lambda left, right: float(left) * float(right),
    "fraction_of_ceiling": lambda value, ceiling: float(value) / float(ceiling),
    "percent": lambda value: 100.0 * float(value),
    # Where an agreement statistic sits between its null and the attainable ceiling: 0 means at
    # the null, 1 means at the ceiling the data itself supports.  A bare p-value cannot say this.
    "null_to_ceiling_position": (
        lambda observed, null, ceiling:
        (float(observed) - float(null)) / (float(ceiling) - float(null))
    ),
}


def resolve_claims(claims: Sequence[Claim], roots: Mapping[str, Path]) -> dict[str, dict]:
    """Resolve every claim against ``roots``, a mapping of prefix to directory.

    A claim's ``file`` begins with a prefix naming its root, for instance
    ``confirm:confirmation/final_labels.json``, so one manifest can cite the confirmatory
    tree, the development tree and the freeze record without ambiguity.
    """

    cache: dict[Path, Any] = {}
    resolved: dict[str, dict] = {}
    for claim in claims:
        if claim.formula is not None:
            derivation = DERIVATIONS.get(claim.formula)
            if derivation is None:
                raise ClaimError(f"claim {claim.key} names unknown derivation {claim.formula!r}")
            missing = [name for name in claim.inputs if name not in resolved]
            if missing:
                raise ClaimError(f"claim {claim.key} needs unresolved inputs {missing}")
            value = derivation(*[resolved[name]["value"] for name in claim.inputs])
            resolved[claim.key] = {
                "value": float(value),
                "kind": claim.kind,
                "digits": claim.digits,
                "note": claim.note,
                "derivation": {"formula": claim.formula, "inputs": list(claim.inputs)},
                "rendered": format_value(value, claim.kind, claim.digits),
            }
            continue

        if claim.file is None:
            raise ClaimError(f"claim {claim.key} has neither a file nor a derivation")
        prefix, _, relative = claim.file.partition(":")
        if prefix not in roots:
            raise ClaimError(f"claim {claim.key} names unknown root {prefix!r}")
        path = roots[prefix] / relative if relative else roots[prefix]
        if path not in cache:
            if not path.exists():
                raise ClaimError(f"claim {claim.key} cites {path}, which does not exist")
            cache[path] = json.loads(path.read_text(encoding="utf-8"))
        try:
            raw = resolve_pointer(cache[path], claim.pointer)
        except ClaimError as error:
            raise ClaimError(f"claim {claim.key} ({claim.file}#{claim.pointer}): {error}") from error
        value = _coerce(raw, claim.kind, claim.key)
        resolved[claim.key] = {
            "value": value,
            "kind": claim.kind,
            "digits": claim.digits,
            "note": claim.note,
            "source": {"file": claim.file, "pointer": claim.pointer},
            "rendered": format_value(value, claim.kind, claim.digits),
        }
    return resolved


def format_value(value: Any, kind: str, digits: int = 4) -> str:
    """Render a value for the page: four significant digits for measured numbers."""

    if kind == "integer":
        return f"{int(value):,}"
    if kind == "boolean":
        return "yes" if value else "no"
    if kind in ("label", "text"):
        return str(value)
    number = float(value)
    if number == 0.0:
        return "0"
    magnitude = abs(number)
    if magnitude >= 10 ** digits or magnitude < 10 ** -(digits + 1):
        return f"{number:.{digits - 1}e}"
    decimals = max(0, digits - 1 - int(f"{magnitude:e}".split("e")[1]))
    return f"{number:.{decimals}f}"
