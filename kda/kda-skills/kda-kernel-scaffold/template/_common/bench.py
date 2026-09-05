"""Kernel timing: CUPTI-backed via ``torch.profiler`` when available, CUDA events otherwise.

Both methods flush L2 before every iteration so a small working set is not served from
cache, and both report the median per-call time in milliseconds. The profiler method sums
only device kernel time, so host launch overhead is excluded; the events method includes it.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

_L2_FLUSH_BYTES = 256 << 20
_flush_buffers: Dict[int, torch.Tensor] = {}


def _flush_l2(device: torch.device) -> None:
    buf = _flush_buffers.get(device.index)
    if buf is None:
        buf = torch.empty(_L2_FLUSH_BYTES, dtype=torch.uint8, device=device)
        _flush_buffers[device.index] = buf
    buf.zero_()


def _current_device(device: Optional[torch.device]) -> torch.device:
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(device)


def _bench_events(fn: Callable[[], Any], warmup: int, iters: int, device: torch.device) -> List[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(iters):
        _flush_l2(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


_MARK = "kda_bench_"


def _bench_profiler(fn: Callable[[], Any], warmup: int, iters: int, device: torch.device) -> List[float]:
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile, record_function

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    # Synchronising after the flush and again inside the annotation makes the annotation's CPU
    # time range bracket exactly the kernels ``fn`` launched, on the profiler's unified clock.
    # Attribution by time range works for Triton launches and autograd-thread kernels alike,
    # where ``device_time_total`` of the annotation would only count aten kernels on this thread.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            _flush_l2(device)
            torch.cuda.synchronize(device)
            with record_function(f"{_MARK}{i}"):
                fn()
                torch.cuda.synchronize(device)
    events = prof.events()
    spans = {e.name: (e.time_range.start, e.time_range.end) for e in events if e.device_type == DeviceType.CPU and e.name.startswith(_MARK)}
    kernels = [e for e in events if e.device_type == DeviceType.CUDA and not e.name.startswith(_MARK)]
    times = []
    for i in range(iters):
        span = spans.get(f"{_MARK}{i}")
        if span is None:
            return []
        start, end = span
        times.append(sum(k.time_range.elapsed_us() for k in kernels if start <= k.time_range.start <= end) / 1e3)
    return times


def bench_ms(
    fn: Callable[[], Any],
    *,
    warmup: int = 5,
    iters: int = 20,
    method: str = "auto",
    device: Optional[torch.device] = None,
) -> float:
    """Median time per ``fn()`` call in ms. ``method`` is ``auto`` | ``profiler`` | ``events``."""
    device = _current_device(device)
    times: List[float] = []
    if method in ("auto", "profiler"):
        try:
            times = _bench_profiler(fn, warmup, iters, device)
        except Exception:
            if method == "profiler":
                raise
        if times and max(times) > 0:
            return statistics.median(times)
        if method == "profiler":
            raise RuntimeError("profiler recorded no device kernels for fn; use method='events'")
    return statistics.median(_bench_events(fn, warmup, iters, device))


def resolve_method(method: str = "auto", device: Optional[torch.device] = None) -> str:
    """The method ``bench_ms(method=...)`` will use on this machine: ``profiler`` or ``events``.

    Reports record this rather than the CLI value, so ``auto`` is never ambiguous in a report.
    """
    if method != "auto":
        return method
    device = _current_device(device)
    try:
        t = torch.ones(1 << 16, device=device)
        times = _bench_profiler(lambda: t.add_(1), 1, 2, device)
        return "profiler" if times and max(times) > 0 else "events"
    except Exception:
        return "events"


def copy_ms(nbytes: float, *, method: str = "auto", device: Optional[torch.device] = None) -> float:
    """Time of a plain device copy moving ``nbytes`` in total (read + write).

    This is the achievable memory roof at that transfer size: HBM peak is approached only
    asymptotically (an A800 copy reaches ~73% of 2039 GB/s at 64 MB, ~84% at 256 MB), so a
    memory-bound kernel is judged against a same-size copy rather than the datasheet number.
    """
    device = _current_device(device)
    n = max(1, int(nbytes // 2))
    src = torch.empty(n, dtype=torch.uint8, device=device)
    dst = torch.empty(n, dtype=torch.uint8, device=device)
    return bench_ms(lambda: dst.copy_(src), method=method, device=device)


def matmul_ms(
    flops: float,
    *,
    shapes: Optional[Sequence[Tuple[int, int, int]]] = None,
    dtype: torch.dtype = torch.bfloat16,
    method: str = "auto",
    device: Optional[torch.device] = None,
) -> Optional[float]:
    """Time of cuBLAS running the phase's GEMMs: the achievable **compute** roof.

    The datasheet tensor-core peak (A800 312 TFLOP/s bf16) is reached by nothing: cuBLAS itself
    lands at 70-85% depending on shape, and ``torch.compile(mode="max-autotune")`` picks cuBLAS
    or a Triton template that ties it. So a tensor-core kernel is judged against cuBLAS on the
    same GEMMs, as a memory-bound kernel is judged against a same-size copy (`copy_ms`).

    ``shapes`` is the list of ``(M, N, K)`` the phase executes (``_speed_of_light.gemm_shapes``);
    each is timed as ``torch.matmul`` on bf16 operands with an fp32 accumulator and the times
    summed. Without shapes a single cube GEMM of the same FLOPs is timed (``n = cbrt(flops / 2)``),
    which cuBLAS runs near its best, so that roof is *tighter* than the real shapes' - give the
    shapes whenever the SPEC knows them. Returns ``None`` for zero FLOPs (a memory-only phase).
    """
    if not flops or flops <= 0:
        return None
    device = _current_device(device)
    if not shapes:
        n = max(16, int(round((flops / 2.0) ** (1.0 / 3.0))))
        n = (n + 15) // 16 * 16
        shapes = [(n, n, n)]
    total = 0.0
    for m, n, k in shapes:
        a = torch.empty(int(m), int(k), dtype=dtype, device=device)
        b = torch.empty(int(n), int(k), dtype=dtype, device=device)
        total += bench_ms(lambda: torch.matmul(a, b.t()), method=method, device=device)
        del a, b
    return total


def configure_compile_for_dev() -> None:
    """Put ``torch.compile`` in a state where a dev run measures the code as it is on disk now.

    Call before any ``torch.compile`` in a verify/bench/probe process (``_run_dev.py`` does; a
    scratch compile probe must too). Two settings:

    - Caches off. Inductor's on-disk FX-graph and AOTAutograd caches are keyed on the graph code
      and input metadata, not on a custom op's *fake* output metadata. After a fake is edited (the
      usual M2 fix), the artefact compiled against the old fake is reused, and the compiled run
      fails with a phantom ``assert_size_stride`` / restride error - or passes for the wrong
      reason. Different kernels registering the same op name (two runs in one container) poison
      each other the same way.
    - No donated buffers. The compiled backward is timed with ``retain_graph=True`` (one graph,
      many calls), which torch >= 2.5 rejects when AOTAutograd donated the saved buffers.
    """
    try:
        import torch._inductor.config as inductor_config

        inductor_config.force_disable_caches = True
    except (ImportError, AttributeError):
        pass
    try:
        import torch._functorch.config as functorch_config

        functorch_config.donated_buffer = False
    except (ImportError, AttributeError):
        pass


def compile_or_none(
    fn: Callable[..., Any], *args: Any, mode: Optional[str] = None, **kwargs: Any
) -> Optional[Callable[..., Any]]:
    """``torch.compile(fn, mode=mode)`` warmed up on ``args``; ``None`` if compilation fails.

    ``mode`` is ``None`` (inductor default) for memory-bound ops and
    ``"max-autotune-no-cudagraphs"`` for tensor-core ops (``_run_dev.py`` picks by
    ``_speed_of_light.ROOF``): with autotuning inductor chooses between cuBLAS and its own
    Triton GEMM templates with the epilogue fused, which is the realistic speed of light for a
    GEMM + epilogue and the number a KDA kernel has to match. CUDA graphs are left off because
    the baseline is timed per call like the kernel.
    """
    configure_compile_for_dev()
    try:
        compiled = torch.compile(fn, mode=mode) if mode else torch.compile(fn)
        compiled(*args, **kwargs)
        return compiled
    except Exception:
        return None


__all__ = ["bench_ms", "copy_ms", "matmul_ms", "compile_or_none", "configure_compile_for_dev", "resolve_method"]
