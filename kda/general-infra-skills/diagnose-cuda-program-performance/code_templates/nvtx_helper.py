"""NVTX annotations for the host layer of a CUDA project."""

## Template: two tokens to replace

# project   ->  your project identifier, lowercase (the NVTX domain name)
# PROJECT   ->  the same identifier, uppercase (the environment switch)
#
# After replacing, `grep -n 'project\|PROJECT' nvtx_helper.py` must come back
# empty.

## Backend resolution

# Resolved once, at import, in this order:
#
#     1. NVIDIA's `nvtx` package  - preferred: carries domains and categories
#     2. torch.cuda.nvtx          - accepted: messages only, no domain/category
#     3. no-op                    - only when PROJECT_NVTX is off/0/false/no
#
# With neither package importable and no waiver set, the import raises
# NvtxBackendUnavailable rather than annotating nothing silently: an empty band
# in a profile is indistinguishable from an idle program, and that ambiguity has
# cost more investigations than a failed import ever will.

## What an NVTX range measures

# The host call, not the GPU work it queues. A range around a launch closes when
# the launch returns, long before the kernel runs. Never synchronize inside a
# range to make its bar match kernel duration: that serializes the program and
# yields a profile unlike the one that ships. Let Nsight Systems correlate the
# range with the GPU work its enclosed launches produced. nvtx_helper.h carries
# the same note for the native layer, with a diagram.

## Where to put ranges

# Treat roughly 1 microsecond as the floor: below that the annotation competes
# with the work it describes. Annotate semantic boundaries, nesting them as
#
#     stage / phase
#       transaction / function
#         selected subphase
#           CUDA API and kernel rows   <- the profiler already names these
#
# Keep messages stable and low-cardinality - "load_batch", not "load_batch 317" -
# so a tool can aggregate across calls. A formatted per-call message costs your
# own time even with no profiler attached, and the preferred backend caches every
# distinct message it is given.

## Domains

# This layer annotates into the domain "project", while the native layer (see
# nvtx_helper.h) uses "project.native". Separate domains keep each layer
# independently filterable in Nsight, at the cost of host and native ranges no
# longer nesting into one visual stack. If a single hierarchy matters more than
# filtering, move both layers to the global NVTX domain: use torch.cuda.nvtx
# directly here, and nvtx3::scoped_range (no _in<>) there.

## Example

#     from nvtx_helper import nvtx_annotate, nvtx_range, profiler_capture
#
#     @nvtx_annotate("load_batch", category="transfer")
#     def load_batch(source):
#         ...
#
#     def run_step(source):
#         with nvtx_range("step"):
#             batch = load_batch(source)
#             with nvtx_range("compute", category="dispatch"):
#                 return compute(batch)
#
#     for index, source in enumerate(sources):
#         # Profile steady state only, with:
#         #   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi ...
#         with profiler_capture(enabled=index == 8):
#             run_step(source)
#
# For work that begins on one thread and finishes on another, use the handle
# form: push/pop is a per-thread stack and cannot span threads.
#
#     handle = nvtx_range_start("request")
#     # ... hand the handle to whichever thread finishes the request ...
#     nvtx_range_end(handle)

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

NVTX_DOMAIN = "project"
"""NVTX domain for this layer; the native layer uses ``project.native``."""

NVTX_CATEGORIES = ("dispatch", "memory", "transfer", "synchronization", "kernel")
"""Starter taxonomy shared with ``nvtx_helper.h``; a convention, not enforced.

Slashes build a hierarchy tools can group on, e.g. ``"kernel/attention"``.
"""

_ENV_SWITCH = "PROJECT_NVTX"
"""Environment variable that waives annotation entirely."""

_ENV_DISABLED_VALUES = frozenset({"0", "off", "false", "no"})
"""Values of ``PROJECT_NVTX`` selecting the no-op backend, case-insensitively."""

_UNAVAILABLE_MESSAGE = (
    "No NVTX backend is available. Pick one:\n"
    "  install a backend  -  pip install nvtx  (preferred: adds domains and "
    "categories), or use a CUDA-enabled PyTorch build, whose torch.cuda.nvtx "
    "bindings this module also accepts\n"
    f"  waive annotation   -  export {_ENV_SWITCH}=off  (ranges become no-ops, "
    "and no host ranges will appear in any profile)"
)
"""Remedies offered when no backend resolves; both change what a profile can attribute."""


class NvtxBackendUnavailable(ImportError):
    """Raised at import when no NVTX backend exists and none was waived."""


## Backends


class _NoOpBackend:
    """Discard every annotation, for runs that waived NVTX."""

    name = "noop"

    def push(self, message: str, category: str | None) -> None:
        """Discard the start of a range."""

    def pop(self) -> None:
        """Discard the end of a range."""

    def start(self, message: str, category: str | None) -> Any:
        """Return a handle standing in for a range that was never opened."""
        return None

    def end(self, handle: Any) -> None:
        """Discard the end of a handle-based range."""

    def mark(self, message: str, category: str | None) -> None:
        """Discard an instantaneous event."""


class _NvtxPackageBackend:
    """Annotate through NVIDIA's ``nvtx`` package, honoring domain and category."""

    name = "nvtx"

    def __init__(self, module: Any) -> None:
        self._nvtx = module

    def push(self, message: str, category: str | None) -> None:
        """Open a range on this thread's stack for ``NVTX_DOMAIN``."""
        self._nvtx.push_range(message=message, domain=NVTX_DOMAIN, category=category)

    def pop(self) -> None:
        """Close the innermost range on this thread's stack."""
        self._nvtx.pop_range(domain=NVTX_DOMAIN)

    def start(self, message: str, category: str | None) -> Any:
        """Open a range closable from any thread, and return its handle."""
        return self._nvtx.start_range(
            message=message, domain=NVTX_DOMAIN, category=category
        )

    def end(self, handle: Any) -> None:
        """Close a range opened by ``start``."""
        self._nvtx.end_range(handle)

    def mark(self, message: str, category: str | None) -> None:
        """Emit an instantaneous event."""
        self._nvtx.mark(message=message, domain=NVTX_DOMAIN, category=category)


class _TorchBackend:
    """Annotate through ``torch.cuda.nvtx``: messages only, global domain.

    PyTorch's bindings expose no domain or category, so both are dropped with one
    warning. Ranges land in the global NVTX domain, which means they do nest with
    native ranges that also use the global domain — see the module's Domains note
    before reading that as either a bug or a feature.
    """

    name = "torch"

    def __init__(self, module: Any) -> None:
        self._nvtx = module.cuda.nvtx
        self._warned = False

    def _warn_dropped(self, category: str | None) -> None:
        """Warn once that this backend records no domain or category."""
        if category is None or self._warned:
            return
        self._warned = True
        warnings.warn(
            "torch.cuda.nvtx cannot record an NVTX domain or category, so "
            f"domain {NVTX_DOMAIN!r} and every category are dropped from this "
            "run's ranges; install the nvtx package to keep them.",
            RuntimeWarning,
            stacklevel=4,
        )

    def push(self, message: str, category: str | None) -> None:
        """Open a range on this thread's stack in the global domain."""
        self._warn_dropped(category)
        self._nvtx.range_push(message)

    def pop(self) -> None:
        """Close the innermost range on this thread's stack."""
        self._nvtx.range_pop()

    def start(self, message: str, category: str | None) -> Any:
        """Open a range closable from any thread, and return its handle."""
        self._warn_dropped(category)
        return self._nvtx.range_start(message)

    def end(self, handle: Any) -> None:
        """Close a range opened by ``start``."""
        self._nvtx.range_end(handle)

    def mark(self, message: str, category: str | None) -> None:
        """Emit an instantaneous event."""
        self._warn_dropped(category)
        self._nvtx.mark(message)


def _annotation_waived() -> bool:
    """Report whether the environment waived annotation for this process."""
    return os.environ.get(_ENV_SWITCH, "").strip().lower() in _ENV_DISABLED_VALUES


def _torch_nvtx_works(module: Any) -> bool:
    """Report whether this PyTorch build carries working NVTX bindings.

    A CPU-only build still exposes ``torch.cuda.nvtx``, but every function there
    is a stub that raises, so pushing and popping one throwaway range is the only
    reliable check.
    """
    try:
        module.cuda.nvtx.range_push("project.nvtx.probe")
        module.cuda.nvtx.range_pop()
    except Exception:  # noqa: BLE001 - any failure means the backend is unusable
        return False
    return True


def _select_backend() -> Any:
    """Resolve the annotation backend once, at import.

    Returns:
        The first usable backend: NVIDIA ``nvtx``, then ``torch.cuda.nvtx``, or
        the no-op backend when annotation was waived.

    Raises:
        NvtxBackendUnavailable: No backend is importable and none was waived.
    """
    if _annotation_waived():
        return _NoOpBackend()

    try:
        import nvtx
    except ImportError:
        pass
    else:
        return _NvtxPackageBackend(nvtx)

    try:
        import torch
    except ImportError:
        pass
    else:
        if _torch_nvtx_works(torch):
            return _TorchBackend(torch)

    raise NvtxBackendUnavailable(_UNAVAILABLE_MESSAGE)


_BACKEND = _select_backend()

NVTX_BACKEND_NAME = _BACKEND.name
"""Which backend resolved: ``nvtx``, ``torch``, or ``noop``.

Worth reporting alongside a profile, since it decides whether domains and
categories were recorded at all.
"""


## Ranges


@contextmanager
def nvtx_range(name: str, *, category: str | None = None) -> Iterator[None]:
    """Annotate one host range, exception-safe and without synchronizing CUDA.

    Args:
        name: Stable, low-cardinality range message.
        category: Category from ``NVTX_CATEGORIES`` or a slash-separated
            refinement of one; ``None`` leaves the range uncategorized. Dropped
            by the ``torch`` backend.

    Yields:
        Nothing; the range closes when the block exits, on any path out.
    """
    _BACKEND.push(name, category)
    try:
        yield
    finally:
        _BACKEND.pop()


def nvtx_annotate(
    name: str, *, category: str | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap one callable in a named NVTX range.

    Args:
        name: Stable, low-cardinality range message.
        category: Category, as in ``nvtx_range``.

    Returns:
        Decorator that preserves the wrapped callable's metadata.
    """

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with nvtx_range(name, category=category):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def nvtx_mark(name: str, *, category: str | None = None) -> None:
    """Emit an instantaneous event rather than an interval.

    Args:
        name: Stable, low-cardinality event message.
        category: Category, as in ``nvtx_range``.
    """
    _BACKEND.mark(name, category)


def nvtx_range_start(name: str, *, category: str | None = None) -> Any:
    """Open a range that a later, possibly different, thread will close.

    Prefer ``nvtx_range`` whenever the range fits a block: push/pop is a
    per-thread stack, so only the handle form may cross threads or outlive a
    lexical scope.

    Args:
        name: Stable, low-cardinality range message.
        category: Category, as in ``nvtx_range``.

    Returns:
        Handle to pass to ``nvtx_range_end``. Opaque: its type differs per
        backend, so do not compare, serialize, or store it across processes.
    """
    return _BACKEND.start(name, category)


def nvtx_range_end(handle: Any) -> None:
    """Close a range opened by ``nvtx_range_start``.

    Args:
        handle: Handle that ``nvtx_range_start`` returned.
    """
    _BACKEND.end(handle)


## Profiler capture window


_CUDART: Any = None
"""Cached ``libcudart`` handle for the profiler-control fallback path."""

# TODO: Windows and future CUDA majors stamp the version into the library name,
# so extend this list if the fallback path fails on your platform.
_CUDART_CANDIDATES = (
    "libcudart.so",
    "libcudart.so.13",
    "libcudart.so.12",
    "libcudart.dylib",
    "cudart64_13.dll",
    "cudart64_12.dll",
)
"""Library names to try, covering installs without an unversioned symlink."""

_PROFILER_UNAVAILABLE_MESSAGE = (
    "Cannot control the CUDA profiler: neither torch.cuda.profiler nor "
    "libcudart could be loaded. Either install a CUDA-enabled PyTorch build, or "
    "drop --capture-range=cudaProfilerApi and trigger on an NVTX range instead: "
    "nsys profile --trace=cuda,nvtx --capture-range=nvtx "
    f"--nvtx-capture='step@{NVTX_DOMAIN}' ..."
)
"""Remedies offered when the capture window cannot be controlled."""


def _load_cudart() -> Any:
    """Load ``libcudart`` for cudaProfilerStart/Stop without importing a framework.

    Returns:
        Cached ``ctypes`` handle to the CUDA runtime library.

    Raises:
        RuntimeError: The library could not be located.
    """
    global _CUDART
    if _CUDART is not None:
        return _CUDART

    import ctypes
    import ctypes.util

    found = ctypes.util.find_library("cudart")
    for candidate in ((found,) if found else ()) + _CUDART_CANDIDATES:
        try:
            _CUDART = ctypes.CDLL(candidate)
        except OSError:
            continue
        return _CUDART

    raise RuntimeError(_PROFILER_UNAVAILABLE_MESSAGE)


def _profiler_control() -> tuple[Callable[[], Any], Callable[[], Any]]:
    """Resolve the pair of functions that start and stop profile collection.

    Returns:
        ``(start, stop)``, from ``torch.cuda.profiler`` when PyTorch is present
        and from ``libcudart`` otherwise.

    Raises:
        RuntimeError: Neither path is available.
    """
    try:
        import torch.cuda.profiler
    except ImportError:
        pass
    else:
        # A CPU-only build imports fine and exposes the module, so probe the
        # attributes rather than the package.
        start = getattr(torch.cuda.profiler, "start", None)
        stop = getattr(torch.cuda.profiler, "stop", None)
        if callable(start) and callable(stop):
            return start, stop

    cudart = _load_cudart()
    return cudart.cudaProfilerStart, cudart.cudaProfilerStop


@contextmanager
def profiler_capture(*, enabled: bool = True) -> Iterator[None]:
    """Bound one profiler capture window, excluding warm-up and teardown.

    Pairs with ``nsys profile --capture-range=cudaProfilerApi``, which then
    collects only what this block encloses. Without a profiler attached under
    that flag it does nothing, so it is safe to leave in place.

    Independent of the NVTX backend on purpose. ``PROJECT_NVTX=off`` waives host
    *annotation*; it does not waive the capture window, and folding the two
    together would silently profile the whole run — including warm-up — for
    anyone who had turned annotation off.

    Args:
        enabled: When false, run the block without touching the profiler — for
            selecting one representative iteration by index.

    Yields:
        Nothing; collection stops when the block exits, on any path out.

    Raises:
        RuntimeError: ``enabled`` is set but no profiler control path exists.
    """
    if not enabled:
        yield
        return

    start, stop = _profiler_control()
    start()
    try:
        yield
    finally:
        stop()
