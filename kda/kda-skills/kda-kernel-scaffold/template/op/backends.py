"""Backend names for `{{op}}`.

One kernel backend per package (SPEC ``kernel_backend``): the kernels live in
``_{{kernel_backend}}/`` and ``KDA_BACKEND`` maps 1:1 onto the names below.
"""

KERNEL_BACKEND = "{{kernel_backend}}"

BACKEND_EAGER = "eager"
BACKEND_KERNEL = KERNEL_BACKEND
# Kernel forward with the backward taken from autograd through the eager reference. If this
# matches the reference and the fused backend does not, the bug is in the backward kernel.
BACKEND_KERNEL_FWD_ONLY = f"{KERNEL_BACKEND}_fwd_only"

BACKENDS = (BACKEND_EAGER, BACKEND_KERNEL, BACKEND_KERNEL_FWD_ONLY)
BACKEND_DEFAULT = BACKEND_KERNEL

# Mirrors SPEC ``recompute``: the scaffold sets these from the M1 decision, and
# ``_run_dev.py`` refuses to run when they disagree with SPEC.md.
RECOMPUTE_AVAILABLE = False
RECOMPUTE_DEFAULT = False

__all__ = [
    "KERNEL_BACKEND",
    "BACKEND_EAGER",
    "BACKEND_KERNEL",
    "BACKEND_KERNEL_FWD_ONLY",
    "BACKENDS",
    "BACKEND_DEFAULT",
    "RECOMPUTE_AVAILABLE",
    "RECOMPUTE_DEFAULT",
]
