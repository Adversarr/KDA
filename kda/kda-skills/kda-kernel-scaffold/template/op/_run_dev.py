"""Verify `{{op}}` against the eager reference and benchmark it; writes report.json + REPORT.md.

Verification always runs, per workload: the training forward (``fwd``, aux written), the
inference forward (``infer``: under ``torch.no_grad()``, ``save_aux=False``), input gradients
(``bwd``) and, when SPEC declares ``recompute.available``, the recompute training path
(``bwd_recompute``: forward without aux, backward recomputing it), all at the tolerances from
SPEC.md. On torch >= 2.6 a ``torch.compile(fullgraph=True)`` probe of the first user workload
catches a fake that does not mirror the launcher. The contract probe (``--contract``, on by
default) feeds that workload a CPU tensor, a non-unit last stride, a shape mismatch and an
integer dtype and requires a Python error before any launch, then checks that zero rows and
``no_grad`` return. A workload whose inputs cannot fit (6x their bytes vs free memory) is
recorded as skipped: required skips are incomplete evidence; optional skips are notes.

``--bench`` adds timing of eager, eager+torch.compile and the kernel for every phase, plus the
speed-of-light estimate from `_speed_of_light.py`.

Run from the repo root as a module, or directly as a script::

    python -m kda_kernels.{{op}}._run_dev --verify
    python kda_kernels/{{op}}/_run_dev.py --verify --bench --backend {{kernel_backend}} --iteration 2

Exit code: pass/tune 0, fail 1, invalid input/SPEC 2, incomplete evidence 3.
"""

if __package__ in (None, ""):  # invoked as a script: re-run as a module so relative imports work
    import runpy
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve()
    sys.path.insert(0, str(_here.parents[2]))
    runpy.run_module(f"{_here.parents[1].name}.{_here.parent.name}._run_dev", run_name="__main__", alter_sys=True)
    raise SystemExit(0)

import argparse
import gc
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .._common import VERSION
from .._common.bench import bench_ms, compile_or_none, configure_compile_for_dev, copy_ms, matmul_ms, resolve_method
from .._common.compat import USE_CUSTOM_OP
from .._common.gpu_info import GpuInfo, compute_ms, get_gpu_info, sol_ms
from .._common.report import ALL_PHASES, RECOMPUTE_PHASE, WorkloadResult, write_report
from .._common.evidence import finalize, output_paths, source_hashes
from .._common.spec import Spec, Workload, load_spec, validate_spec
from .._common.verify import compare, grads, lowest_precision
from . import _speed_of_light as sol
from ._dispatch import dispatch
from .backends import (
    BACKEND_DEFAULT,
    BACKEND_EAGER,
    BACKEND_KERNEL_FWD_ONLY,
    BACKENDS,
    KERNEL_BACKEND,
    RECOMPUTE_AVAILABLE,
    RECOMPUTE_DEFAULT,
)
from .interface import {{op}}

OP_DIR = Path(__file__).resolve().parent
# Inputs, outputs, gradients, saved tensors and the baselines' copies: a conservative multiple
# of the input bytes a workload needs to run every phase.
MEMORY_FIT_FACTOR = 6
# Tensor-core ops (`ROOF = "bf16"`/`"fp16"` in _speed_of_light.py): the compute roof is measured
# cuBLAS on the phase's GEMMs and the compile baseline is autotuned; see `_common/bench.py`.
TENSOR_CORE_ROOF = getattr(sol, "ROOF", "fp32") in ("bf16", "fp16")
COMPILE_MODE = "max-autotune-no-cudagraphs" if TENSOR_CORE_ROOF else None


def _gemm_shapes(phase: str, base: Dict[str, torch.Tensor], params: Dict) -> Optional[List[Tuple[int, int, int]]]:
    """``_speed_of_light.gemm_shapes(phase, **inputs)`` when the roofline defines it, else ``None``."""
    fn = getattr(sol, "gemm_shapes", None)
    if fn is None:
        return None
    try:
        shapes = fn(phase, **base, **params)
    except NotImplementedError:
        return None
    return [tuple(int(d) for d in s) for s in shapes] if shapes else None


def make_inputs(w: Workload, device: torch.device) -> Dict[str, torch.Tensor]:
    """Inputs for one workload at the model's scales (the default is unscaled ``randn`` for every input).

    Numerics and the verdict depend on these scales: norm weights ``1 + 0.1 * randn``, gates in
    (0, 1), positions as integers, GEMM weights at the model's init std (``0.02 * randn``:
    ``randn`` on both operands gives ``|z| ~ sqrt(K)``, which no trained model produces and
    which makes every residual epilogue cancel).
    """
    inputs = w.make_inputs(device, seed=0)
    # TODO(scaffold): e.g. inputs["weight"] = 1 + 0.1 * inputs["weight"]  (norm) or *= 0.02 (Linear)
    return inputs


def _call(
    backend: str, tensors: Dict[str, torch.Tensor], params: Dict, recompute: Optional[bool] = None
) -> Tuple[torch.Tensor, ...]:
    extra = {} if recompute is None else {"recompute": recompute}
    return tuple({{op}}(**tensors, **params, backend=backend, **extra))


def _leaves(w: Workload, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        k: v.detach().clone().requires_grad_(w.needs_grad(k) and v.is_floating_point())
        for k, v in tensors.items()
    }


def _compare(spec: Spec, w: Workload, actual: torch.Tensor, expected: torch.Tensor, reduced_over: int = 1) -> dict:
    # Tolerance of the lowest precision in the chain: an fp32 weight grad of a bf16 workload
    # inherits bf16 rounding from the eager reference's intermediates.
    atol, rtol = spec.tolerance(lowest_precision(w.torch_dtype, expected.dtype))
    return compare(actual, expected, atol=atol, rtol=rtol, reduced_over=reduced_over).as_dict()


def _grad_outputs(outputs: Sequence[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(1)
    return [torch.randn(o.shape, dtype=o.dtype, device=device, generator=gen) for o in outputs]


def _bench_phase(
    r: WorkloadResult,
    label: str,
    fwd: Callable[..., Tuple[torch.Tensor, ...]],
    w: Workload,
    base: Dict[str, torch.Tensor],
    device: torch.device,
    method: str,
) -> None:
    """Time ``infer`` (no_grad), ``fwd`` (training forward) and ``bwd`` of ``fwd`` under ``label``."""
    names = list(base)
    plain = [base[k].detach() for k in names]
    with torch.no_grad():
        r.time_ms[f"{label}_infer"] = bench_ms(lambda: fwd(*plain), method=method, device=device)

    leaves = _leaves(w, base)
    wrt = [leaves[k] for k in names if leaves[k].requires_grad]
    if not wrt:
        r.time_ms[f"{label}_fwd"] = r.time_ms[f"{label}_infer"]
        return
    args = [leaves[k] for k in names]
    # Training forward: grad mode on and leaves requiring grad, so the kernel writes its aux.
    r.time_ms[f"{label}_fwd"] = bench_ms(lambda: fwd(*args), method=method, device=device)
    outputs = fwd(*args)
    gouts = _grad_outputs(outputs, device)
    r.time_ms[f"{label}_bwd"] = bench_ms(
        lambda: torch.autograd.grad(outputs, wrt, gouts, retain_graph=True), method=method, device=device
    )


def _bench_recompute(
    r: WorkloadResult,
    fwd_rc: Callable[..., Tuple[torch.Tensor, ...]],
    w: Workload,
    base: Dict[str, torch.Tensor],
    device: torch.device,
    method: str,
) -> None:
    """Time the backward of the recompute path (``kernel_bwd_recompute``); its baseline is the eager backward."""
    names = list(base)
    leaves = _leaves(w, base)
    wrt = [leaves[k] for k in names if leaves[k].requires_grad]
    if not wrt:
        return
    outputs = fwd_rc(*[leaves[k] for k in names])
    gouts = _grad_outputs(outputs, device)
    r.time_ms[f"kernel_{RECOMPUTE_PHASE}"] = bench_ms(
        lambda: torch.autograd.grad(outputs, wrt, gouts, retain_graph=True), method=method, device=device
    )


def _compare_grads(
    spec: Spec, w: Workload, grad_names: List[str], ker_g: Sequence, ref_g: Sequence, ref_out: Sequence[torch.Tensor]
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    largest_output = max(o.numel() for o in ref_out)
    for name, e, k in zip(grad_names, ref_g, ker_g):
        if e is None and k is None:
            continue
        e = torch.zeros_like(k) if e is None else e
        k = torch.zeros_like(e) if k is None else k
        # A gradient smaller than the output was reduced over ~(output / grad) rows.
        reduced_over = max(1, largest_output // max(1, e.numel()))
        out[name] = _compare(spec, w, k, e, reduced_over=reduced_over)
    return out


def _compile_probe(
    r: WorkloadResult,
    w: Workload,
    backend: str,
    base: Dict[str, torch.Tensor],
    ref_out: Tuple[torch.Tensor, ...],
    ref_g: Sequence,
    grad_names: List[str],
    spec: Spec,
    device: torch.device,
) -> None:
    """``torch.compile(fullgraph=True)`` forward and backward through the backend.

    A fake that does not mirror the launcher (shape, dtype, rank of an aux output), a graph
    break or a stride guard shows up here, at M2, instead of in the user's compiled training
    loop. Goes through ``_dispatch`` so the environment lookup in ``interface`` is not traced.
    """
    names = list(base)

    def fn(*ts: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(dispatch(backend, **dict(zip(names, ts)), **w.params))

    leaves = _leaves(w, base)
    configure_compile_for_dev()  # caches off: a stale artefact from an older fake would be reused
    try:
        out = torch.compile(fn, fullgraph=True)(*[leaves[k] for k in names])
        r.compile = {f"out{i}": _compare(spec, w, k, e) for i, (k, e) in enumerate(zip(out, ref_out))}
        if grad_names:
            ker_g = grads(out, [leaves[k] for k in grad_names], _grad_outputs(ref_out, device))
            r.compile.update(_compare_grads(spec, w, grad_names, ker_g, ref_g, ref_out))
    except Exception as exc:  # noqa: BLE001 - recorded as the verdict reason
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        r.compile_error = f"{type(exc).__name__}: {first_line[:300]}"
    finally:
        torch._dynamo.reset()


def _memory_fit(w: Workload, device: torch.device) -> Optional[str]:
    """``None`` when the workload should fit, else the skip note recorded in the report."""
    need = MEMORY_FIT_FACTOR * w.input_bytes()
    # Release the previous row first: its baselines' tensors and the compiled graph's buffers sit
    # in the caching allocator (and dynamo's guards) after `run_workload` returns, and
    # `mem_get_info` counts reserved-but-unused blocks as taken, which can incorrectly
    # skip later rows unless those reservations are released.
    gc.collect()
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(device)
    if need <= free:
        return None
    return f"needs {need / 2**30:.1f} GiB ({MEMORY_FIT_FACTOR}x inputs), free {free / 2**30:.1f} GiB"


def run_workload(
    w: Workload,
    backend: str,
    device: torch.device,
    spec: Spec,
    info: GpuInfo,
    do_bench: bool,
    method: str,
    probe_compile: bool = False,
) -> WorkloadResult:
    r = WorkloadResult(name=w.name, required=w.required, dtype=w.dtype)
    r.skipped = _memory_fit(w, device)
    if r.skipped:
        return r
    with_recompute = spec.recompute.available and backend != BACKEND_EAGER
    measuring = False
    try:
        base = make_inputs(w, device)
        ref_in, ker_in = _leaves(w, base), _leaves(w, base)
        ref_out = _call(BACKEND_EAGER, ref_in, w.params)
        ker_out = _call(backend, ker_in, w.params)
        if len(ref_out) != len(ker_out):
            raise RuntimeError(f"backend returned {len(ker_out)} outputs, reference {len(ref_out)}")
        r.fwd = {f"out{i}": _compare(spec, w, k, e) for i, (k, e) in enumerate(zip(ker_out, ref_out))}
        # Inference path: under no_grad the kernel skips its saved-for-backward stores
        # (``save_aux=False``); the user-visible outputs must be identical.
        with torch.no_grad():
            inf_out = _call(backend, {k: v.detach() for k, v in base.items()}, w.params)
        if len(inf_out) != len(ref_out):
            raise RuntimeError("inference output count differs from the reference")
        r.infer = {f"out{i}": _compare(spec, w, k, e) for i, (k, e) in enumerate(zip(inf_out, ref_out))}

        grad_names = [k for k in base if ref_in[k].requires_grad]
        ref_g: Sequence = []
        ker_g: Sequence = []
        gouts: Sequence = []
        rc_in = rc_out = rc_g = None
        if grad_names:
            gouts = _grad_outputs(ref_out, device)
            ref_g = grads(ref_out, [ref_in[k] for k in grad_names], gouts)
            ker_g = grads(ker_out, [ker_in[k] for k in grad_names], gouts)
            r.bwd = _compare_grads(spec, w, grad_names, ker_g, ref_g, ref_out)

        if with_recompute:
            # Recompute training path: forward without aux, backward recomputing it.
            rc_in = _leaves(w, base)
            rc_out = _call(backend, rc_in, w.params, recompute=True)
            if len(rc_out) != len(ref_out):
                raise RuntimeError("recompute output count differs from the reference")
            r.bwd_recompute = {f"out{i}": _compare(spec, w, k, e) for i, (k, e) in enumerate(zip(rc_out, ref_out))}
            if grad_names:
                rc_g = grads(rc_out, [rc_in[k] for k in grad_names], _grad_outputs(ref_out, device))
                r.bwd_recompute.update(_compare_grads(spec, w, grad_names, rc_g, ref_g, ref_out))

        if probe_compile:
            _compile_probe(r, w, backend, base, ref_out, ref_g, grad_names, spec, device)

        # The verification tensors (two sets of leaves with their graphs, outputs, grads) are
        # ~15x the inputs on a 2^31-element row; the bench needs only `base`. Free them first.
        del ref_in, ker_in, ref_out, ker_out, inf_out, ref_g, ker_g, gouts, rc_in, rc_out, rc_g
        torch.cuda.empty_cache()

        if do_bench:
            measuring = True
            names = list(base)

            def eager_fn(*ts):
                return _call(BACKEND_EAGER, dict(zip(names, ts)), w.params)

            def kernel_fn(*ts):
                return _call(backend, dict(zip(names, ts)), w.params)

            def kernel_rc_fn(*ts):
                return _call(backend, dict(zip(names, ts)), w.params, recompute=True)

            _bench_phase(r, "eager", eager_fn, w, base, device, method)
            # The compiled baseline is optional: if it fails to build or to time, the gate falls
            # back to eager and records why, instead of losing the kernel measurement. Tensor-core
            # ops get inductor's autotuned GEMM (cuBLAS or a Triton template with the epilogue
            # fused): that is the realistic speed of light for a GEMM + epilogue.
            compiled = compile_or_none(eager_fn, *[base[k].detach() for k in names], mode=COMPILE_MODE)
            if compiled is None:
                r.warnings.append("torch.compile baseline unavailable (compilation failed); baseline is eager")
            else:
                try:
                    _bench_phase(r, "compiled", compiled, w, base, device, method)
                except Exception as exc:  # noqa: BLE001 - any baseline failure is non-fatal
                    for phase in ALL_PHASES:
                        r.time_ms.pop(f"compiled_{phase}", None)
                    r.warnings.append(f"torch.compile baseline failed to time ({type(exc).__name__}: {exc}); baseline is eager")
            _bench_phase(r, "kernel", kernel_fn, w, base, device, method)
            if with_recompute:
                _bench_recompute(r, kernel_rc_fn, w, base, device, method)
            measuring = False
            for phase in ALL_PHASES:
                if f"kernel_{phase}" not in r.time_ms:
                    continue
                try:
                    nbytes, flops = sol.estimate(phase, **base, **w.params)
                except NotImplementedError:
                    r.sol_ms[phase] = r.roof_ms[phase] = None
                    continue
                r.sol_ms[phase] = sol_ms(nbytes, flops, info, roof=sol.ROOF)
                # Achievable roof: a same-size copy (memory) vs the compute roof, whichever is
                # slower. On tensor cores the compute roof is cuBLAS on the phase's GEMMs
                # (`gemm_shapes` when the roofline names them, a same-FLOP cube otherwise), not
                # the datasheet peak nothing reaches; on CUDA cores it stays the datasheet number.
                copy = copy_ms(nbytes, method=method, device=device)
                if TENSOR_CORE_ROOF:
                    shapes = _gemm_shapes(phase, base, w.params)
                    compute = matmul_ms(flops, shapes=shapes, dtype=w.torch_dtype, method=method, device=device)
                else:
                    compute = compute_ms(flops, info, roof=sol.ROOF)
                r.roof_ms[phase] = copy if compute is None else max(copy, compute)
    except torch.cuda.OutOfMemoryError as exc:
        # The 6x-inputs estimate is a floor: an fp32-upcast eager reference of a 2^31-element
        # norm holds ~10 temporaries of 8 GiB, and a shared card has less free memory than the
        # probe saw. An optional row records the OOM as a skip (it proves nothing either way);
        # a required row is incomplete; preserve any already-observed numerical failures.
        torch.cuda.empty_cache()
        note = f"out of memory while running it ({str(exc).splitlines()[0][:120]}); the {MEMORY_FIT_FACTOR}x-inputs fit estimate was too low"
        r.skipped = note
        if not w.required:
            r.fwd = r.infer = r.bwd = r.bwd_recompute = None
            r.time_ms = {}
    except Exception as exc:  # recorded in the report; the gate decides
        if measuring:
            r.warnings.append(f"benchmark evidence unavailable: {type(exc).__name__}: {exc}")
        else:
            r.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return r


# ---------------------------------------------------------------- contract probe


def _run_case(fn: Callable[[], object], expect: str) -> Tuple[bool, str]:
    """Run one mutation; ``expect`` is ``"raise"`` (a Python ValueError/TypeError before any
    launch) or ``"return"``. A CUDA-side error surfacing at synchronize counts as a launch."""
    try:
        fn()
        torch.cuda.synchronize()
    except (ValueError, TypeError) as exc:
        first = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else ""
        return expect == "raise", f"raised {type(exc).__name__}: {first}"
    except Exception as exc:  # noqa: BLE001 - a launch happened or the check is not a contract check
        first = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else ""
        return False, f"raised {type(exc).__name__} instead of ValueError/TypeError: {first}"
    return expect == "return", ("returned" if expect == "return" else "accepted a bad input")


def contract_probe(w: Workload, backend: str, device: torch.device, row_inputs: Optional[List[str]] = None) -> Dict[str, dict]:
    """Mutations of the workload's first tensor input that `interface.py` must reject or accept.

    Rows: ``cpu_tensor``, ``bad_last_stride`` (same shape, last stride 2), ``shape_mismatch``
    (last dim narrowed by one; only when another input shares that dim), ``bad_dtype`` (int32)
    must raise; ``zero_rows`` (leading dim 0 on the SPEC ``row_inputs``, or on every input
    sharing the first input's leading dim when SPEC leaves it null) and ``no_grad`` must
    return. Run last: a rejected-by-CUDA input can poison the context.
    """
    base = make_inputs(w, device)
    names = [k for k, v in base.items() if isinstance(v, torch.Tensor)]
    first = names[0]
    x = base[first]
    rows: Dict[str, dict] = {}

    def case(name: str, expect: str, inputs: Dict[str, torch.Tensor], no_grad: bool = False) -> None:
        def fn() -> object:
            if no_grad:
                with torch.no_grad():
                    return _call(backend, inputs, w.params)
            return _call(backend, inputs, w.params)

        passed, detail = _run_case(fn, expect)
        rows[name] = {"expect": expect, "passed": passed, "detail": detail}

    case("cpu_tensor", "raise", dict(base, **{first: x.cpu()}))
    if x.dim() >= 1 and x.shape[-1] >= 2:
        wide = torch.randn((*x.shape[:-1], 2 * x.shape[-1]), dtype=x.dtype, device=device)
        case("bad_last_stride", "raise", dict(base, **{first: wide[..., ::2]}))
        shares_last_dim = any(base[n].dim() >= 1 and base[n].shape[-1] == x.shape[-1] for n in names[1:])
        if shares_last_dim:
            case("shape_mismatch", "raise", dict(base, **{first: x[..., :-1]}))
    case("bad_dtype", "raise", dict(base, **{first: x.to(torch.int32)}))
    if x.dim() >= 2:
        if row_inputs is not None:
            zero = {n: base[n][:0] for n in row_inputs if n in base}
        else:
            zero = {n: base[n][:0] for n in names if base[n].dim() >= 2 and base[n].shape[0] == x.shape[0]}
        case("zero_rows", "return", dict(base, **zero))
    case("no_grad", "return", dict(base), no_grad=True)
    return rows


# ---------------------------------------------------------------- main


def _consistency_errors(spec: Spec) -> List[str]:
    """SPEC.md and backends.py must agree on the kernel backend and the recompute policy."""
    errors = []
    if spec.kernel_backend != KERNEL_BACKEND:
        errors.append(f"SPEC kernel_backend {spec.kernel_backend!r} != backends.KERNEL_BACKEND {KERNEL_BACKEND!r}")
    if spec.recompute.available != RECOMPUTE_AVAILABLE:
        errors.append(f"SPEC recompute.available {spec.recompute.available} != backends.RECOMPUTE_AVAILABLE {RECOMPUTE_AVAILABLE}")
    if spec.recompute.default != RECOMPUTE_DEFAULT:
        errors.append(f"SPEC recompute.default {spec.recompute.default} != backends.RECOMPUTE_DEFAULT {RECOMPUTE_DEFAULT}")
    return errors


def _kernel_backend_version() -> Optional[str]:
    try:
        mod = __import__(KERNEL_BACKEND)
    except ImportError:
        return None
    return getattr(mod, "__version__", None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify", action="store_true", help="accepted for readability; verification always runs")
    p.add_argument("--bench", action="store_true", help="also time eager / compiled / kernel per phase and compute SOL")
    p.add_argument("--backend", default=BACKEND_DEFAULT, choices=BACKENDS)
    p.add_argument("--workload", action="append", help="run only these workload names (repeatable)")
    p.add_argument("--method", default="auto", choices=("auto", "profiler", "events"), help="timing method")
    p.add_argument("--device", default="cuda")
    p.add_argument("--iteration", type=int, default=None, help="unique dispatch run number from STATUS, recorded in the report")
    p.add_argument("--json", help="JSON output (scoped runs require a diagnostic path)")
    p.add_argument("--md", help="Markdown output (scoped runs require a diagnostic path)")
    p.add_argument("--finalize-audit", type=Path, help="finalize current reports from an audit JSON file without GPU work")
    p.add_argument("--contract", dest="contract", action="store_true", default=True, help="contract probe (default on)")
    p.add_argument("--no-contract", dest="contract", action="store_false", help="skip the contract probe")
    p.add_argument(
        "--no-compile-probe",
        action="store_true",
        help="skip the torch.compile(fullgraph=True) probe of the first user workload (custom-op torch only)",
    )
    args = p.parse_args(argv)
    try:
        args.json, args.md = output_paths(OP_DIR, bool(args.workload), args.json, args.md)
        if args.finalize_audit:
            if args.bench or args.workload or args.no_compile_probe or not args.contract:
                p.error("--finalize-audit cannot be combined with measurement/probe options")
            report = finalize(OP_DIR, args.finalize_audit)
            decision = report["verification"]["verdict"]
            print(f"finalized verification: {decision}")
            return {"fail": 1, "incomplete": 3}.get(decision, 0)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    spec = load_spec(OP_DIR / "SPEC.md")
    errors = validate_spec(spec) + _consistency_errors(spec)
    if errors:
        print("SPEC.md is invalid:\n  - " + "\n  - ".join(errors), file=sys.stderr)
        return 2

    selected = [w for w in spec.workloads if not args.workload or w.name in args.workload]
    if not selected or (args.workload and set(args.workload) - {w.name for w in selected}):
        print(f"unknown workload selection: {args.workload}", file=sys.stderr)
        return 2
    hashes_before = source_hashes(OP_DIR)
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    info = get_gpu_info(device)

    # One compile probe and one contract probe per run, on the user's workload (the shapes a
    # training loop sees). The compile probe is not for the eager reference nor the *_fwd_only
    # bisecting backend (an autograd.Function by design).
    users = ([w for w in selected if w.required and w.source == "user"]
             or [w for w in selected if w.required]
             or selected)
    probe_on = None
    if USE_CUSTOM_OP and args.backend not in (BACKEND_EAGER, BACKEND_KERNEL_FWD_ONLY) and not args.no_compile_probe:
        probe_on = users[0].name
    results = [
        run_workload(w, args.backend, device, spec, info, args.bench, args.method, probe_compile=(w.name == probe_on))
        for w in selected
    ]
    if args.contract:
        target = next(r for r in results if r.name == users[0].name)
        if not target.skipped:
            try:
                target.contract = contract_probe(users[0], args.backend, device, spec.row_inputs)
            except Exception as exc:  # noqa: BLE001 - a broken probe must not hide the run
                target.contract = {"probe": {"expect": "return", "passed": False, "detail": f"{type(exc).__name__}: {exc}"}}

    env = {
        "gpu": info.name,
        "cc": info.cc,
        "torch": torch.__version__,
        "kernel_backend": KERNEL_BACKEND,
        "kernel_backend_version": _kernel_backend_version(),
        "common_version": VERSION,
        "bench_method": resolve_method(args.method, device) if args.bench else None,  # resolved, never "auto"
        "peaks": info.as_dict(),
    }
    coverage = {"all_workloads": [w.name for w in spec.workloads], "benchmark": args.bench, "required": {}}
    for w in selected:
        if not w.required:
            continue
        phases = ["fwd", "infer"]
        if any(w.needs_grad(name) for name in w.shapes):
            phases.append("bwd")
        if spec.recompute.available and args.backend != BACKEND_EAGER:
            phases.append("bwd_recompute")
        timings = []
        if args.bench:
            timings = [f"kernel_{phase}" for phase in phases if phase != "bwd_recompute" or "bwd" in phases]
            timings += [f"eager_{phase}" for phase in phases if phase != "bwd_recompute"]
        coverage["required"][w.name] = {
            "phases": phases, "timings": timings, "contract": w.name == users[0].name,
            "compile": USE_CUSTOM_OP and args.backend not in (BACKEND_EAGER, BACKEND_KERNEL_FWD_ONLY) and w.name == users[0].name,
        }
    report = write_report(
        args.json,
        args.md,
        op=spec.op,
        backend=args.backend,
        env=env,
        results=results,
        iteration=args.iteration,
        coverage=coverage,
        source_hashes=hashes_before,
        gate_recompute=spec.recompute.default,
        kernel_backend=KERNEL_BACKEND,
        # An explicit --method events run is a host-overhead diagnostic: its times include launch
        # overhead, so SOL/baseline do not gate it (auto resolving to events on a CUPTI-less machine still gates).
        gate_perf=args.method != "events",
    )

    print(f"[{spec.op}/{args.backend}] verdict: {report['verdict']}")
    for r in report["workloads"]:
        line = f"  {r['name']:<24}"
        if r.get("skipped"):
            print(line + f" skipped: {r['skipped']}")
            continue
        for phase in ALL_PHASES:
            cell = r.get(phase)
            if cell is None and phase == RECOMPUTE_PHASE:
                continue
            ok = "-" if cell is None else ("ok" if all(v["passed"] for v in cell.values()) else "FAIL")
            line += f" {phase}={ok:<4}"
        if r.get("compile_error"):
            line += " compile=FAIL"
        elif r.get("compile") is not None:
            line += f" compile={'ok' if all(v['passed'] for v in r['compile'].values()) else 'FAIL'}"
        if r.get("contract") is not None:
            n_ok = sum(v["passed"] for v in r["contract"].values())
            line += f" contract={n_ok}/{len(r['contract'])}"
        for phase in ALL_PHASES:
            d = r["derived"].get(phase)
            if d:
                eff = "n/a" if d["sol_eff"] is None else f"{d['sol_eff']:.2f}"
                spd = "n/a" if d["speedup"] is None else f"{d['speedup']:.2f}x"
                line += f"  {phase}: {r['time_ms'][f'kernel_{phase}']:.4f} ms, sol_eff={eff}, speedup={spd}"
        if r["error"]:
            line += f"  error: {r['error']}"
        print(line)
    for reason in report["reasons"]:
        print(f"  - {reason}")
    print(f"  wrote {args.json} and {args.md}")
    return {"fail": 1, "incomplete": 3}.get(report["verdict"], 0)


if __name__ == "__main__":
    sys.exit(main())
