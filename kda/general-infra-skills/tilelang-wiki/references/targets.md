# Targets And Execution Backends

`target` chooses the **compiler** (CUDA, HIP, Metal, LLVM, …).
`execution_backend` chooses the **runtime adapter** that launches the
compiled kernel (`tvm_ffi`, `cython`, `nvrtc`, `cutedsl`, Metal `torch`).
They are independent. Auto-resolve happens only when a value is `None` or
`"auto"`.

Defaults come from:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TILELANG_DEFAULT_TARGET` | `auto` | String (`cuda`) or JSON dict (`{"kind":"cuda","arch":"sm_100"}`) |
| `TILELANG_EXECUTION_BACKEND` | `auto` | First **available** backend registered for that target |
| `TILELANG_VERBOSE` | `0` | Log requested vs resolved backend |

`TILELANG_TARGET` is gone. `execution_backend="dlpack"` is gone. CUDA
`"torch"` is gone. Metal still registers a `torch` backend.

```python
kernel = tilelang.compile(func, target="cuda", execution_backend="tvm_ffi")
kernel = tilelang.compile(func, target={"kind": "cuda", "arch": "sm_90"})
```

## Auto Detect

`target="auto"` walks registered detectors in import order and takes the
first available device:

1. **CUDA** — `nvcc` / CUDA home present and `torch.version.hip` is `None`.
   Arch comes from `torch.cuda.get_device_capability` when CUDA is visible.
2. **HIP** — ROCm path present.
3. **Metal** — macOS arm64.

CPU / LLVM is **not** auto-detected. Ask for it explicitly:
`target="llvm"` or `target="c"`.

A current TVM `Target` context, if any, wins over auto-detect.

## Execution Backends By Target

| Target | Registered backends (first available wins for `"auto"`) |
| --- | --- |
| `cuda` | `tvm_ffi`, `nvrtc` (if available), `cython`. CuteDSL is a separate target key. |
| `hip` | `tvm_ffi`, `cython` |
| `llvm` / `c` | `tvm_ffi` (llvm), `cython` then `tvm_ffi` (`c`) |
| `metal` | `tvm_ffi`, `torch` |
| `webgpu` | `cython`, `tvm_ffi` (emit-only; see below) |

`resolve_execution_backend("auto", target)` picks the first registered
backend whose `is_available()` is true. Asking for an unregistered name
raises with the allowed list. So `"auto"` is `tvm_ffi` on CUDA even when
`cuda-python` (which makes `nvrtc` available) is installed; pick `nvrtc`
by name if you want it.

### Which CUDA backend, measured (A800, TileLang 0.1.13 and 0.1.14)

The backend is the launcher, not the compiler: the generated kernel is the
same and its **device time is identical** under all three (bf16 GEMM
128x128x32: 0.330 / 0.332 ms; FA2 causal forward: 2.01 / 2.15 / 2.02 ms, the
spread is clock drift). They differ in two costs:

| backend | cold compile of one kernel | host launch overhead per call | breaks on |
| --- | --- | --- | --- |
| `tvm_ffi` (auto) | 5.7 s = TVM passes 2.2 + nvcc 2.7 + host `cc` 0.8 | **10 us** (cheaper than a Triton launch, 25 us) | - |
| `nvrtc` | **4.4 s** = TVM passes 2.2 + NVRTC 2.0 (no nvcc, no host cc) | 21 us (pure-Python `cuLaunchKernelEx`) | a tensor argument named `res`, `config`, `kernels` or `stream`: the generated launcher shadows it with its own local (`'CUresult' object has no attribute 'data_ptr'`) |
| `cython` | 11 s (compiles a Cython wrapper) | 27 us | `T.StridedTensor` with `T.dynamic` extents (`Cannot use and / or / not operator to Expr`) |

Rules that follow: `nvrtc` when compile time dominates (many shapes, an
autotune sweep, verification over a dozen workloads) and the kernels
are long; `tvm_ffi` when many short kernels are launched per step (a
training loop with 10 us kernels feels the 11 us difference). Neither, nor
any `pass_configs` key (`TL_ENABLE_FAST_MATH`, `TL_PTXAS_REGISTER_USAGE_LEVEL`,
`TL_DISABLE_SAFE_MEMORY_ACCESS`, measured on the GEMM: 0.332 -> 0.337 /
0.341 / 0.332 ms), changes what the kernel does on the device. The TileLang
autotuner runs candidates in worker processes and needs `tvm_ffi` there
(`tuner.set_compile_args(execution_backend="tvm_ffi")`); compile the winner
with whichever backend you ship.

What does move the compile bill is what is `T.const`: every distinct value
is a new 4-6 s compile. Extents that only enter the grid and base offsets
(batch, heads, GEMM rows `M`) cost nothing as `T.dynamic` (GEMM 0.323 ms
dynamic vs 0.331 static; FA2 forward 2.09 vs 2.13); extents that set loop
trip counts or tile predicates (`S` of an attention, `N`, `K`, `D`) do
(dynamic `S` on the FA2 forward: +13% device time, +50% compile time). Keep
those `T.const`.

## CUDA And CuteDSL

Ordinary CUDA uses `target="cuda"` or `{"kind": "cuda", "arch": "sm_90"}`.
CuteDSL is the same CUDA kind with a `cutedsl` key:

```python
@tilelang.jit(target="cutedsl", execution_backend="cutedsl")
```

Requires `nvidia-cutlass-dsl>=4.3.1,!=4.3.4` and an importable
`cutlass.cute`. Version `4.3.4` is banned. The adapter emits a Python
executor (`.py`), not a C wrapper.

## LLVM / CPU

`target="llvm"` (or `"c"`) is runnable through `execution_backend="tvm_ffi"`.
Use the same `T.Kernel` launch; backends without SIMT ignore thread extents.
There is no `is_cpu=` flag.

## Metal

`target="metal"` auto-detects on arm64 macOS. Apple M5+ with macOS / SDK 26+
adds a `metal4` key and cooperative-tensor GEMM
(`metal.cooperative_tensor` scope). Older Apple Silicon stays simdgroup.
Default Metal backend is `tvm_ffi`; `torch` is also registered.

## WebGPU

`target="webgpu"` is **emit-only**. Codegen produces a `WebGPUModule` /
source; it is not a first-class run path in this skill. Prefer CUDA / HIP /
Metal / LLVM for kernels you intend to execute.

## Practical Rules

- Pin `target=` in examples that must be reproducible across machines.
- Do not pass `execution_backend="dlpack"` or CUDA `execution_backend="torch"`.
- JSON `TILELANG_DEFAULT_TARGET` must be real JSON object syntax, not
  Python dict literals (`{kind: "cuda"}` fails).
- See `tilelang/env/basic.md` for cache interaction with target/backend
  keys, and `debug.md` when lowering differs across targets.
