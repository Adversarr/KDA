"""Backend and recompute selection with process-wide kill switches.

``KDA_BACKEND`` in the environment overrides every per-call ``backend=`` argument, so a
researcher debugging a divergence can force ``KDA_BACKEND=eager`` without touching code.
``KDA_RECOMPUTE`` does the same for the recompute training path (``1``/``true`` on,
``0``/``false`` off), for ops whose SPEC declares ``recompute.available``.
"""

import os
from typing import Optional, Sequence

ENV_BACKEND = "KDA_BACKEND"
ENV_RECOMPUTE = "KDA_RECOMPUTE"
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def resolve_backend(requested: Optional[str], default: str, allowed: Sequence[str]) -> str:
    """Return the backend to run: env override, else ``requested``, else ``default``."""
    chosen = os.environ.get(ENV_BACKEND) or requested or default
    if chosen not in allowed:
        raise ValueError(
            f"Unknown backend {chosen!r} (from {ENV_BACKEND} or backend=); allowed: {list(allowed)}"
        )
    return chosen


def resolve_recompute(requested: Optional[bool], default: bool) -> bool:
    """Return whether the backward recomputes aux: env override, else ``requested``, else ``default``."""
    raw = os.environ.get(ENV_RECOMPUTE)
    if raw is not None and raw.strip() != "":
        value = raw.strip().lower()
        if value in _TRUE:
            return True
        if value in _FALSE:
            return False
        raise ValueError(f"{ENV_RECOMPUTE}={raw!r} is not a boolean (use 1/0)")
    if requested is not None:
        return bool(requested)
    return bool(default)


__all__ = ["ENV_BACKEND", "ENV_RECOMPUTE", "resolve_backend", "resolve_recompute"]
