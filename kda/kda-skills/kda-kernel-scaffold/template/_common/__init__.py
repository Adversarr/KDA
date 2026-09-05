"""Shared runtime vendored into a user repo by the KDA scaffold.

Every kernel package under ``kda_kernels/`` imports from here. This package has no
dependency on KDA itself; it is copied once per repo and versioned by ``VERSION``.
"""

from pathlib import Path

VERSION = (Path(__file__).parent / "VERSION").read_text().strip()

__all__ = ["VERSION"]
