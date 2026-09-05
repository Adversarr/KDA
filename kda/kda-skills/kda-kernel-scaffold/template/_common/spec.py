"""Load the machine-readable part of ``SPEC.md``.

``SPEC.md`` opens with a YAML front matter block delimited by ``---`` lines; everything
after it is prose for humans and agents. ``_run_dev.py`` reads only the front matter, so
workloads, tolerances and the backward policy have a single source of truth.

Front matter fields (see the scaffold template for a filled example)::

    kda_spec: 2
    op: <snake_case op name>
    common_version: <_common/VERSION this op was generated against>
    source: {path, symbol}
    kernel_backend: triton | nvmath | tilelang
    compute_pattern: <a name from PATTERNS, mirrored by reference/common/compute-patterns.md>
    backward: fused | eager
    backward_reason: <required when backward == eager>
    compute_dtype: fp32 | bf16 | fp16
    compute_dtype_source: user | inferred | default
    recompute: {available: bool, default: bool, why: str | null}
    row_inputs: [arg, ...] | null   # inputs whose leading dim is the token/row dim (zero_rows probe)
    target_archs: [sm80, sm90, sm100]
    tolerances: {bf16: [atol, rtol], ...}          # optional per-dtype overrides
    saved_for_backward: [{name, shape, dtype, bytes, why}]
    workloads: [{name, source: user|representative|edge, required, dtype, shapes: {arg: [dims]},
                 strides: {arg: [strides]}, dtypes: {arg: fp32|bool|int32|int64}, grad_inputs: [arg, ...], params: {}}]
    hardware: [<GpuInfo.as_dict()>]
    integration: {definition_site, flag: {env, config}, recompute_flag: {env, config}, smoke_command}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .verify import DTYPES, TOLERANCES

# Non-float inputs a workload may declare in `dtypes` (masks, positions, index tables). They
# never take a gradient and never set the workload's tolerance; `make_inputs` fills a bool with a
# mostly-true mask and an int with small non-negative indices, and an op whose table has meaning
# (positions on a grid, an `inv_freq` table) overrides them in `_run_dev.make_inputs`.
INDEX_DTYPES: Dict[str, torch.dtype] = {"bool": torch.bool, "int32": torch.int32, "int64": torch.int64}

SPEC_VERSION = 2
FRONT_MATTER_DELIM = "---"
WORKLOAD_SOURCES = ("user", "representative", "edge")
BACKWARD_MODES = ("fused", "eager")
KERNEL_BACKENDS = ("triton", "nvmath", "tilelang")  # nvmath = cuBLASLt through nvmath-python (GEMM + epilogue, no kernel); tilelang experimental
COMPUTE_DTYPE_SOURCES = ("user", "inferred", "default")
# Mirrored, row for row, by kda-kernel-implement/reference/common/compute-patterns.md.
PATTERNS = (
    "elementwise_stream",
    "rowwise_onepass",
    "rope_pairs",
    "permute_copy",
    "gemm_tensorcore",
    "flash_attention_2",
    "swiglu_dual_gemm",
)
# Where int32 element offsets overflow: the reason the `edge_huge` row exists, and its size.
INT32_ELEMENTS = 2**31
# Inputs a workload may declare. The harness budgets 6x the input bytes (inputs, outputs, grads,
# saved tensors, the baselines' copies), so ~13 GiB is what an idle 80 GB card can host, and
# 13 GiB is exactly a bf16 activation plus an fp32 stream at 2^31 elements (6 bytes x 2^31); a
# row above it is never run, only "skipped", and proves nothing. `edge_huge` needs one tensor
# of 2^31+1 elements (4 GiB in bf16), not the largest shape that fits.
MAX_WORKLOAD_INPUT_BYTES = 13 * 2**30
# An `edge_huge` row whose inputs are all below 2^31 elements must be small (the crossing is then
# a logical plane, e.g. attention scores B*H*Sq*Skv); a big row that crosses nothing is a mistake.
EDGE_HUGE_SMALL_INPUT_BYTES = 2 * 2**30


@dataclass
class Recompute:
    available: bool = False
    default: bool = False
    why: Optional[str] = None


@dataclass
class Workload:
    name: str
    source: str
    required: bool
    dtype: str  # default dtype of every tensor input
    shapes: Dict[str, List[int]]
    # Optional explicit strides per input (same rank as the shape, last stride 1). An input
    # without an entry is contiguous. This is how the strided workloads of reference/workloads.md
    # are expressed; `make_inputs` builds a view over a larger buffer.
    strides: Dict[str, List[int]] = field(default_factory=dict)
    dtypes: Dict[str, str] = field(default_factory=dict)  # per-input overrides, e.g. {weight: fp32, key_valid: bool}
    grad_inputs: Optional[List[str]] = None  # inputs that receive gradients; None = all
    params: Dict[str, Any] = field(default_factory=dict)  # non-tensor kwargs, e.g. {eps: 1.0e-6}

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPES[self.dtype]

    def dtype_of(self, name: str) -> torch.dtype:
        dt = self.dtypes.get(name, self.dtype)
        return DTYPES[dt] if dt in DTYPES else INDEX_DTYPES[dt]

    def needs_grad(self, name: str) -> bool:
        if not self.dtype_of(name).is_floating_point:
            return False
        return self.grad_inputs is None or name in self.grad_inputs

    def storage_numel(self, name: str) -> int:
        """Elements the backing buffer of ``name`` needs (larger than ``numel`` when strided)."""
        shape = self.shapes[name]
        strides = self.strides.get(name)
        if not strides:
            return _prod(shape)
        if any(s == 0 for s in shape):
            return 0
        return 1 + sum((s - 1) * st for s, st in zip(shape, strides))

    def input_bytes(self) -> int:
        """Bytes of every input's backing storage; the memory-fit estimate starts from this."""
        return sum(self.storage_numel(n) * self.dtype_of(n).itemsize for n in self.shapes)

    def make_inputs(self, device: torch.device, seed: int = 0) -> Dict[str, torch.Tensor]:
        """Standard-normal inputs for every entry of ``shapes``; ops override in ``_run_dev.py``.

        A strided input is a view over a buffer of ``storage_numel`` elements, so the kernel
        sees exactly the strides SPEC declares (a padded row, a slice of a fused qkv buffer).
        """
        gen = torch.Generator(device=device).manual_seed(seed)

        def fill(n: int, dtype: torch.dtype) -> torch.Tensor:
            if dtype is torch.bool:  # a mask with ~10% false, so masked paths run and rows stay non-empty
                return torch.rand(n, device=device, generator=gen) >= 0.1
            if not dtype.is_floating_point:  # indices; an op with a real table overrides in _run_dev
                return torch.randint(0, 1024, (n,), dtype=dtype, device=device, generator=gen)
            return torch.randn(n, dtype=dtype, device=device, generator=gen)

        out: Dict[str, torch.Tensor] = {}
        for name, shape in self.shapes.items():
            dtype = self.dtype_of(name)
            strides = self.strides.get(name)
            if not strides:
                out[name] = fill(_prod(shape), dtype).reshape(tuple(shape))
                continue
            out[name] = fill(self.storage_numel(name), dtype).as_strided(tuple(shape), tuple(strides))
        return out


@dataclass
class Spec:
    op: str
    backward: str
    workloads: List[Workload]
    tolerances: Dict[torch.dtype, Tuple[float, float]]
    raw: Dict[str, Any]
    kernel_backend: str = "triton"
    compute_pattern: str = ""
    compute_dtype: str = "fp32"
    compute_dtype_source: str = "default"
    recompute: Recompute = field(default_factory=Recompute)
    # Inputs whose leading dim is the token/row dim; the harness's `zero_rows` contract probe
    # empties exactly these. None = every input sharing the first input's leading dim, which
    # is wrong when a parameter happens to match it (a GEMM weight (N, K) with N == M).
    row_inputs: Optional[List[str]] = None

    def tolerance(self, dtype: torch.dtype) -> Tuple[float, float]:
        return self.tolerances.get(dtype, TOLERANCES.get(dtype, TOLERANCES[torch.float32]))

    @property
    def required_workloads(self) -> List[Workload]:
        return [w for w in self.workloads if w.required]


def _prod(xs: List[int]) -> int:
    n = 1
    for x in xs:
        n *= int(x)
    return n


def parse_front_matter(text: str) -> Dict[str, Any]:
    """Return the YAML mapping between the leading ``---`` delimiters of ``text``."""
    import yaml

    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONT_MATTER_DELIM:
        raise ValueError("SPEC.md must start with a '---' YAML front matter block")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == FRONT_MATTER_DELIM)
    except StopIteration:
        raise ValueError("SPEC.md front matter is not closed by a '---' line") from None
    data = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(data, dict):
        raise ValueError("SPEC.md front matter must be a YAML mapping")
    return data


_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _coerce_param(value: Any) -> Any:
    """YAML 1.1 reads ``1e-6`` (no dot) as a string; kernels want the float."""
    if isinstance(value, str) and _NUMBER.match(value.strip()):
        text = value.strip()
        return int(text) if re.fullmatch(r"[+-]?\d+", text) else float(text)
    return value


def spec_from_dict(data: Dict[str, Any]) -> Spec:
    tolerances = {
        DTYPES[name]: (float(pair[0]), float(pair[1]))
        for name, pair in (data.get("tolerances") or {}).items()
        if name in DTYPES
    }
    workloads = [
        Workload(
            name=str(w["name"]),
            source=str(w.get("source", "representative")),
            required=bool(w.get("required", True)),
            dtype=str(w["dtype"]),
            shapes={k: [int(x) for x in v] for k, v in (w.get("shapes") or {}).items()},
            strides={k: [int(x) for x in v] for k, v in (w.get("strides") or {}).items()},
            dtypes={k: str(v) for k, v in (w.get("dtypes") or {}).items()},
            grad_inputs=list(w["grad_inputs"]) if w.get("grad_inputs") is not None else None,
            params={k: _coerce_param(v) for k, v in (w.get("params") or {}).items()},
        )
        for w in (data.get("workloads") or [])
    ]
    rc = data.get("recompute") or {}
    recompute = Recompute(
        available=bool(rc.get("available", False)),
        default=bool(rc.get("default", False)),
        why=None if rc.get("why") in (None, "") else str(rc.get("why")),
    )
    return Spec(
        op=str(data.get("op", "")),
        backward=str(data.get("backward", "fused")),
        workloads=workloads,
        tolerances=tolerances,
        raw=data,
        kernel_backend=str(data.get("kernel_backend", "triton")),
        compute_pattern=str(data.get("compute_pattern") or ""),
        compute_dtype=str(data.get("compute_dtype", "fp32")),
        compute_dtype_source=str(data.get("compute_dtype_source", "default")),
        recompute=recompute,
        row_inputs=list(data["row_inputs"]) if data.get("row_inputs") is not None else None,
    )


def load_spec(path: Union[str, Path]) -> Spec:
    return spec_from_dict(parse_front_matter(Path(path).read_text()))


def _stride_errors(w: Workload) -> List[str]:
    """Strides must match the shape's rank, end in 1 and describe a non-overlapping view."""
    errors: List[str] = []
    for name, strides in w.strides.items():
        if name not in w.shapes:
            errors.append(f"workload {w.name!r}: strides for unknown input {name!r}")
            continue
        shape = w.shapes[name]
        if len(strides) != len(shape):
            errors.append(f"workload {w.name!r}: strides of {name!r} have rank {len(strides)}, shape has {len(shape)}")
            continue
        if not strides or strides[-1] != 1:
            errors.append(f"workload {w.name!r}: last stride of {name!r} must be 1 (last dim contiguous)")
            continue
        if any(s <= 0 for s in strides):
            errors.append(f"workload {w.name!r}: strides of {name!r} must be positive")
            continue
        # No overlap: sorted by stride, every dim must clear the extent of the faster ones.
        dims = sorted(range(len(shape)), key=lambda i: strides[i])
        for a, b in zip(dims, dims[1:]):
            if shape[a] > 1 and strides[b] < shape[a] * strides[a]:
                errors.append(f"workload {w.name!r}: strides of {name!r} overlap (dim {b} stride {strides[b]} < {shape[a]}*{strides[a]})")
                break
    return errors


def _size_errors(w: Workload) -> List[str]:
    """Rows must be runnable, and the ultra-large row must cross 2^31 elements, not fill the card.

    Oversized `edge_huge` rows can be skipped without exercising the int32 overflow they
    exist for: 2^31 elements is 4 GiB of bf16 or 8 GiB of fp32.
    """
    errors: List[str] = []
    if not w.shapes or any(name not in w.shapes for name in w.strides):
        return errors
    try:
        total = w.input_bytes()
        biggest = max(_prod(shape) for shape in w.shapes.values())
    except (KeyError, TypeError, ValueError):
        return errors
    if total > MAX_WORKLOAD_INPUT_BYTES:
        errors.append(
            f"workload {w.name!r}: inputs are {total / 2**30:.1f} GiB; the harness needs 6x that, which no 80 GB card "
            f"has (limit {MAX_WORKLOAD_INPUT_BYTES / 2**30:.0f} GiB of inputs). An edge row is the smallest shape with one "
            f"tensor above 2^31 elements ({INT32_ELEMENTS + 1} elements = 4 GiB bf16), see workloads.md"
        )
    if w.source == "edge" and w.name.startswith("edge_huge") and biggest <= INT32_ELEMENTS and total > EDGE_HUGE_SMALL_INPUT_BYTES:
        errors.append(
            f"workload {w.name!r}: {total / 2**30:.1f} GiB of inputs but no tensor above 2^31 elements (largest "
            f"{biggest}); the row exists for int32 offset overflow, so give one tensor {INT32_ELEMENTS + 1}+ elements "
            f"(and shrink the rest), or keep it under {EDGE_HUGE_SMALL_INPUT_BYTES / 2**30:.0f} GiB when the crossing is a logical plane"
        )
    return errors


def validate_spec(spec: Spec) -> List[str]:
    """Human-readable problems that must be fixed before M1 can be approved."""
    errors: List[str] = []
    raw = spec.raw
    if not spec.op:
        errors.append("missing 'op'")
    if raw.get("kda_spec") != SPEC_VERSION:
        errors.append(f"'kda_spec' must be {SPEC_VERSION}")
    if not raw.get("common_version"):
        errors.append("missing 'common_version'")
    if spec.kernel_backend not in KERNEL_BACKENDS:
        errors.append(f"'kernel_backend' must be one of {KERNEL_BACKENDS}")
    if spec.compute_pattern not in PATTERNS:
        errors.append(f"'compute_pattern' must be one of {PATTERNS} (see compute-patterns.md)")
    if spec.compute_dtype not in DTYPES:
        errors.append(f"'compute_dtype' must be one of {list(DTYPES)}")
    if spec.compute_dtype_source not in COMPUTE_DTYPE_SOURCES:
        errors.append(f"'compute_dtype_source' must be one of {COMPUTE_DTYPE_SOURCES}")
    if spec.recompute.default and not spec.recompute.available:
        errors.append("'recompute.default: true' requires 'recompute.available: true'")
    if spec.recompute.available and not spec.recompute.why:
        errors.append("'recompute.available: true' requires 'recompute.why'")
    if spec.backward not in BACKWARD_MODES:
        errors.append(f"'backward' must be one of {BACKWARD_MODES}")
    if spec.backward == "eager" and not raw.get("backward_reason"):
        errors.append("'backward: eager' requires 'backward_reason'")
    if not spec.workloads:
        errors.append("no workloads")
    if spec.workloads and not spec.required_workloads:
        errors.append("at least one workload must be required")
    for w in spec.workloads:
        if w.dtype not in DTYPES:
            errors.append(f"workload {w.name!r}: dtype {w.dtype!r} not in {list(DTYPES)}")
        if w.source not in WORKLOAD_SOURCES:
            errors.append(f"workload {w.name!r}: source {w.source!r} not in {WORKLOAD_SOURCES}")
        if not w.shapes:
            errors.append(f"workload {w.name!r}: no shapes")
        for name, dt in w.dtypes.items():
            if name not in w.shapes or (dt not in DTYPES and dt not in INDEX_DTYPES):
                errors.append(f"workload {w.name!r}: bad dtypes entry {name}: {dt} (float {list(DTYPES)} or {list(INDEX_DTYPES)})")
        for name in w.grad_inputs or []:
            if name in w.dtypes and w.dtypes[name] in INDEX_DTYPES:
                errors.append(f"workload {w.name!r}: grad_inputs names non-float input {name!r}")
            if name not in w.shapes:
                errors.append(f"workload {w.name!r}: grad_inputs names unknown input {name!r}")
        errors += _stride_errors(w)
        errors += _size_errors(w)
    if not any(w.source == "user" for w in spec.workloads):
        errors.append("no workload with source 'user' (the user's real shapes matter most)")
    for name in spec.row_inputs or []:
        if not all(name in w.shapes for w in spec.workloads):
            errors.append(f"row_inputs names {name!r}, which is not an input of every workload")
    for key in ("source", "integration"):
        if not isinstance(raw.get(key), dict):
            errors.append(f"missing '{key}' mapping")
    return errors


__all__ = [
    "SPEC_VERSION",
    "KERNEL_BACKENDS",
    "PATTERNS",
    "COMPUTE_DTYPE_SOURCES",
    "INT32_ELEMENTS",
    "MAX_WORKLOAD_INPUT_BYTES",
    "Recompute",
    "Workload",
    "Spec",
    "parse_front_matter",
    "spec_from_dict",
    "load_spec",
    "validate_spec",
]
