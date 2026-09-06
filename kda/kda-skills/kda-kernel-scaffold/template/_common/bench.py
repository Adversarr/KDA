"""Kernel timing: CUPTI-backed via ``torch.profiler`` when available, CUDA events otherwise.

Both methods flush L2 before every iteration so a small working set is not served from
cache, and both report the median per-call time in milliseconds. The profiler method sums
only device kernel time, so host launch overhead is excluded; the events method includes it.
"""

from __future__ import annotations

import statistics
import math
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


class _IncompleteProfilerTrace(RuntimeError):
    """Missing/invalid collection evidence, distinct from an exception in the kernel."""
    def __init__(self, details):
        self.details = details
        super().__init__(details["reason"])


_MEASUREMENT_DIAGNOSTICS: List[dict] = []


def measurement_diagnostics() -> List[dict]:
    """Copy failed profiler-attempt evidence, including the bounded retry outcome."""
    import copy
    return copy.deepcopy(_MEASUREMENT_DIAGNOSTICS)


def _raw_profiler_samples(events, iters: int):
    """Use original CPU-span matching without building PyTorch CPU event trees.

    Kineto timestamps are nanoseconds on one trace clock. Synthetic GPU user
    annotations and hidden events are excluded, just as in parsed-event timing.
    """
    from torch.autograd import DeviceType

    if iters <= 0:
        raise ValueError("profiler iterations must be positive")
    spans, kernels = {}, []
    cpu_count = cuda_count = 0
    invalid = []
    for event in events:
        device_type, name = event.device_type(), event.name()
        if device_type == DeviceType.CPU:
            cpu_count += 1
        elif device_type == DeviceType.CUDA:
            cuda_count += 1
        if getattr(event, "is_hidden_event", lambda: False)():
            continue
        marker = device_type == DeviceType.CPU and name.startswith(_MARK)
        kernel = device_type == DeviceType.CUDA and not name.startswith(_MARK)
        if not (marker or kernel):
            continue
        start, end = event.start_ns(), event.end_ns()
        try:
            valid_timestamps = math.isfinite(start) and math.isfinite(end) and end >= start
        except (TypeError, ValueError, OverflowError):
            valid_timestamps = False
        if not valid_timestamps:
            invalid.append("nonfinite or reversed event timestamps")
            continue
        if marker:
            if name in spans:
                invalid.append("duplicate iteration annotation")
            spans[name] = (start, end)
        else:
            kernels.append((start, end))
    times, counts = [], []
    for i in range(iters):
        span = spans.get(f"{_MARK}{i}")
        if span is None:
            invalid.append("missing iteration annotation")
            times.append(0.0)
            counts.append(0)
            continue
        start, end = span
        durations = [ke - ks for ks, ke in kernels if start <= ks <= end]
        times.append(sum(durations) / 1e6)
        counts.append(len(durations))
    if len(spans) != iters:
        invalid.append("unexpected annotation count")
    if any(not math.isfinite(t) or t <= 0 for t in times):
        invalid.append("missing positive finite device timing for an iteration")
    details = {"cpu_events": cpu_count, "cuda_events": cuda_count,
               "annotations": len(spans), "matched_kernels": counts,
               "samples_ms": [t if math.isfinite(t) else None for t in times]}
    if invalid:
        details["reason"] = "; ".join(dict.fromkeys(invalid))
        raise _IncompleteProfilerTrace(details)
    return times


def _bench_profiler(fn: Callable[[], Any], warmup: int, iters: int, device: torch.device) -> List[float]:
    from torch.profiler import ProfilerActivity, profile, record_function

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            _flush_l2(device)
            torch.cuda.synchronize(device)
            with record_function(f"{_MARK}{i}"):
                fn()
                torch.cuda.synchronize(device)
    # prof.events() constructs millions of CPU FunctionEvents for chunked eager
    # baselines. Only raw GPU durations and the ten CPU annotations are needed.
    result = getattr(getattr(prof, "profiler", None), "kineto_results", None)
    try:
        if result is None:
            raise _IncompleteProfilerTrace({"reason": "Kineto trace unavailable"})
        return _raw_profiler_samples(result.events() or [], iters)
    except _IncompleteProfilerTrace as exc:
        # Explicit opt-in only: normal reports contain counts/reasons, not paths.
        import os
        directory = os.environ.get("KDA_PROFILER_DIAGNOSTICS_DIR")
        if directory:
            from pathlib import Path
            from uuid import uuid4
            try:
                target = Path(directory) / f"failed-profiler-{uuid4().hex}.trace.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(target))
                exc.details["chrome_trace"] = str(target)
            except Exception as export_error:
                exc.details["trace_export_error"] = type(export_error).__name__
        raise


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
        retry_record = None
        for attempt in range(2):
            try:
                times = _bench_profiler(fn, warmup, iters, device)
                if len(times) != iters or any(not math.isfinite(t) or t <= 0 for t in times):
                    raise _IncompleteProfilerTrace({"reason": "incomplete profiler samples",
                        "samples_ms": [t if math.isfinite(t) else None for t in times]})
            except _IncompleteProfilerTrace as exc:
                record = dict(exc.details, method="profiler", attempt=attempt + 1,
                              callable=getattr(fn, "__qualname__", type(fn).__name__),
                              warmup=warmup, iters=iters,
                              outcome="retry_pending" if attempt == 0 else "retry_exhausted")
                _MEASUREMENT_DIAGNOSTICS.append(record)
                if attempt == 0:
                    retry_record = record
                    continue
                if retry_record is not None:
                    retry_record["outcome"] = "retry_failed"
                if method == "auto":
                    record["fallback"] = "events"
                    break
                raise
            except Exception:
                if retry_record is not None:
                    retry_record["outcome"] = "retry_runtime_error"
                if method == "profiler":
                    raise
                if retry_record is not None:
                    retry_record["fallback"] = "events"
                break
            else:
                if retry_record is not None:
                    retry_record["outcome"] = "retry_succeeded"
                return statistics.median(times)

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



def _attention_samples(q_shape, k_shape, v_shape, dtype, method, device):
    """Measure independent dense Flash phases; no candidate or masked baseline is used."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    generator = torch.Generator(device=device).manual_seed(1729)
    q, k, v = [torch.randn(shape, dtype=dtype, device=device, generator=generator,
                           requires_grad=True) for shape in (q_shape, k_shape, v_shape)]
    options = {"dropout_p": 0.0, "is_causal": False}
    if q_shape[1] != k_shape[1]:
        options["enable_gqa"] = True
    # The context is held across warmup and every timed closure; unsupported Flash
    # raises instead of silently selecting a math or memory-efficient backend.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        def forward():
            return scaled_dot_product_attention(q, k, v, **options)

        def inference():
            with torch.no_grad():
                return forward()

        with torch.enable_grad():
            output = forward()
            upstream = torch.randn(output.shape, dtype=dtype, device=device, generator=generator)

            def backward():
                return torch.autograd.grad(output, (q, k, v), upstream, retain_graph=True)

            closures = {"infer": inference, "fwd": forward, "bwd": backward}
            samples = {phase: [] for phase in closures}
            for _ in range(3):
                for phase, fn in closures.items():
                    samples[phase].append(bench_ms(fn, warmup=3, iters=10,
                                                   method=method, device=device))
    return samples


_ATTENTION_CACHE: Dict[tuple, dict] = {}


def attention_ms(
    phase: str, *, q_shape, k_shape, v_shape, pairs: float,
    dtype: torch.dtype = torch.bfloat16, method: str = "profiler",
    device: Optional[torch.device] = None,
) -> dict:
    """Dense Flash-calibrated useful-work time, with explicit availability/provenance.

    Density is allowed query/key pairs divided by the dense plane across all query
    heads and batches. Scaling is a throughput proxy, not an exact sparse latency
    model or a claim of equivalent intermediate rounding. Callers combine it with
    the compulsory-byte copy floor. Unsupported calibration never falls back.
    """
    import math
    import operator

    if phase not in ("infer", "fwd", "bwd", "bwd_recompute"):
        raise ValueError("unknown attention phase")
    if method not in ("auto", "profiler", "events"):
        raise ValueError("unknown benchmark method")
    shapes = []
    for shape in (q_shape, k_shape, v_shape):
        if len(shape) != 4 or any(isinstance(d, bool) for d in shape):
            raise ValueError("attention shapes must be four positive integers (B,H,S,D)")
        try:
            shape = tuple(operator.index(d) for d in shape)
        except TypeError as exc:
            raise ValueError("attention dimensions must be integers") from exc
        if any(d <= 0 for d in shape):
            raise ValueError("attention dimensions must be positive")
        shapes.append(shape)
    q_shape, k_shape, v_shape = shapes
    b, hq, sq, d = q_shape
    if (k_shape[0] != b or v_shape[0] != b or k_shape[1:3] != v_shape[1:3]
            or k_shape[3] != d or hq % k_shape[1]):
        raise ValueError("incompatible attention batch, heads, sequence or Q/K dimensions")
    dense_pairs = b * hq * sq * k_shape[2]
    if isinstance(pairs, bool) or not math.isfinite(float(pairs)) or not 0 <= pairs <= dense_pairs:
        raise ValueError("pairs must be finite and between zero and the dense pair count")
    density = float(pairs) / dense_pairs
    result = {"method": "attention", "available": False, "roof_ms": None,
              "phase": phase, "density": density, "pairs": float(pairs),
              "dense_pairs": dense_pairs, "q_shape": list(q_shape),
              "k_shape": list(k_shape), "v_shape": list(v_shape),
              "dtype": str(dtype), "backend": "FLASH_ATTENTION", "samples": {},
              "precision": "storage-dtype calibration; intermediate rounding may differ"}
    if dtype not in (torch.bfloat16, torch.float16):
        result["reason"] = "Flash calibration requires bf16 or fp16 storage"
        return result
    try:
        device = _current_device(device)
        result["device"] = str(device)
        if device.type != "cuda":
            raise RuntimeError("Flash calibration requires a CUDA device")
        resolved_method = resolve_method(method, device)
        result["bench_method"] = resolved_method
        key = (q_shape, k_shape, v_shape, dtype, str(device), resolved_method)
        samples = _ATTENTION_CACHE.get(key)
        if samples is None:
            samples = _attention_samples(q_shape, k_shape, v_shape, dtype, resolved_method, device)
            if any(len(samples[p]) != 3 or any(not math.isfinite(t) or t <= 0 for t in samples[p])
                   for p in ("infer", "fwd", "bwd")):
                raise RuntimeError("Flash calibration returned invalid timing samples")
            if len(_ATTENTION_CACHE) >= 16:
                _ATTENTION_CACHE.pop(next(iter(_ATTENTION_CACHE)))
            _ATTENTION_CACHE[key] = samples
        result["samples"] = {p: list(values) for p, values in samples.items()}
        # Recompute owns a fresh forward plus backward, each using the existing
        # minimum-of-three round-median convention.
        dense_ms = (min(samples["fwd"]) + min(samples["bwd"]) if phase == "bwd_recompute"
                    else min(samples[phase]))
        result.update(available=True, dense_ms=dense_ms, roof_ms=density * dense_ms)
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result

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


__all__ = ["measurement_diagnostics", "attention_ms", "bench_ms", "copy_ms", "matmul_ms", "compile_or_none", "configure_compile_for_dev", "resolve_method"]
