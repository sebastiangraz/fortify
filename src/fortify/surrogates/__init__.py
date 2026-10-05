"""Surrogate registry. Names are what FORTIFY_SURROGATES and --surrogates take."""

from __future__ import annotations

from .base import Context, Ensemble, Surrogate

# name → (factory, default weight). Imports are lazy so a missing optional dep only
# breaks the surrogate that needs it.
DEFAULT_WEIGHTS = {"lama": 1.0, "florence2": 0.05, "sam": 0.5}


def make(name: str) -> Surrogate:
    if name == "lama":
        from .lama import LamaSurrogate

        return LamaSurrogate()
    if name == "florence2":
        from .florence2 import Florence2Surrogate

        return Florence2Surrogate()
    if name == "sam":
        from .sam import SamSurrogate

        return SamSurrogate()
    if name in ("toy", "null"):
        from .toy import NullSurrogate, ToySurrogate

        return ToySurrogate() if name == "toy" else NullSurrogate()
    raise ValueError(f"unknown surrogate {name!r}; known: {', '.join(DEFAULT_WEIGHTS)}")


def ensemble(spec: str) -> Ensemble:
    """'lama,sam' or 'lama:1,sam:0.3' → Ensemble."""
    members = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, weight = part.partition(":")
        members.append((make(name), float(weight) if weight else DEFAULT_WEIGHTS.get(name, 1.0)))
    if not members:
        raise ValueError("no surrogates given")
    return Ensemble(members)


__all__ = ["DEFAULT_WEIGHTS", "Context", "Ensemble", "Surrogate", "ensemble", "make"]
