# Environment Basics

TileLang exposes a process-wide environment object as `tilelang.env`. Most
users interact with it indirectly through top-level helpers and environment
variables.

```python
import tilelang

tilelang.disable_cache()
assert not tilelang.is_cache_enabled()
tilelang.enable_cache()

print(tilelang.env.get_default_target())
print(tilelang.env.get_default_execution_backend())
```

## Common APIs

| API | Purpose |
| --- | --- |
| `tilelang.env` | Global `Environment` object. Many fields read from `os.environ`. |
| `tilelang.enable_cache()` | Re-enable TileLang kernel caching in the current process. |
| `tilelang.disable_cache()` | Disable TileLang kernel caching in the current process. |
| `tilelang.is_cache_enabled()` | Return whether kernel caching is currently enabled. |

There is **no** public `tilelang.clear_cache()`. The on-disk kernel cache is
cleared only by deleting a specific directory under `TILELANG_CACHE_DIR`, or
by bypassing cache with `TILELANG_DISABLE_CACHE=1`.

`tilelang.disable_cache()` is a runtime switch. It is useful when debugging
whether a compile path is hitting cache.

`TILELANG_DISABLE_CACHE=1` is stronger than `tilelang.enable_cache()`.
`is_cache_enabled()` returns `False` when either the runtime switch is
disabled or `TILELANG_DISABLE_CACHE` is truthy.

Truthy values are case-insensitive: `1`, `true`, `yes`, `on`.

## Cache Directories

`TILELANG_CACHE_DIR` is the root for kernel cache files. The default is
`~/.tilelang/cache`. `TILELANG_TMP_DIR` defaults to a `tmp` subdirectory
under that root.

Compiled kernels are namespaced as:

```text
$TILELANG_CACHE_DIR/<version>/<os-arch>/kernels/<key>
```

For example `~/.tilelang/cache/0.1.13/darwin-arm64/kernels/...`. Version
local labels such as `+cu128` are sanitized into the version directory name.

Cache keys include the lowered function script, output indices, specialization
arguments, target, host target, execution backend, pass configs, compile flags,
TileLang version, and platform. On Darwin the key also includes the Torch
version.

Native-library content hashing is **opt-in**:

```text
TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1
```

When set, a SHA-256 stamp of `libtilelang` / TVM runtime libraries is folded
into the key so C++ pass changes invalidate cache without a version bump.

Changing the kernel body, target, backend, pass configs, or compile flags
should produce a different key. Editing unrelated Python scaffolding may not.

## Clearing Or Bypassing Cache

1. Set `TILELANG_DISABLE_CACHE=1` to bypass kernel cache globally.
2. Call `tilelang.disable_cache()` before compiling in the current process.
3. Manually remove a specific namespaced cache directory only when you accept
   the risk.

Autotune has a separate disk-cache switch:

```text
TILELANG_AUTO_TUNING_DISABLE_CACHE=1
```

For a full fresh autotune debug run, use both switches. A `ref_prog`,
`supply_prog`, or `manual_check_prog` callback also disables autotune persist
(no stable identity for closures).

## Target, Backend, And Verbose Defaults

If callers do not pass explicit compile arguments, TileLang reads defaults
from the environment:

| Variable | Default | Used for |
| --- | --- | --- |
| `TILELANG_DEFAULT_TARGET` | `auto` | Default compilation target. JSON dicts allowed. |
| `TILELANG_EXECUTION_BACKEND` | `auto` | Default runtime execution backend. |
| `TILELANG_VERBOSE` | `0` | Verbose compilation and backend-resolution logs. |

`TILELANG_TARGET` and `TILELANG_CLEAR_CACHE` are gone. Use
`TILELANG_DEFAULT_TARGET` and the cache-bypass flags above.

JSON target configs parse through `json.loads`. Example:

```bash
TILELANG_DEFAULT_TARGET='{"kind": "cuda", "arch": "sm_100"}' TILELANG_VERBOSE=1 python run_kernel.py
```

A bare string such as `cuda` is also accepted. Explicit `target=` /
`execution_backend=` arguments override these defaults.

## Compile Logging And Temporary Files

`TILELANG_PRINT_ON_COMPILATION` controls high-level compile start/end messages
(default on). `TILELANG_VERBOSE` controls lower-level compile/cache/backend
logs. Temporary compiler files are cleaned by default
(`TILELANG_CLEANUP_TEMP_FILES=1`). HIP builds also expose
`TILELANG_HIP_SAVE_TEMP_FILES=0`.

## CUDA And ROCm Discovery

`tilelang.env.CUDA_HOME` and `tilelang.env.ROCM_HOME` are detected during
environment initialization. CUDA checks `CUDA_HOME`/`CUDA_PATH`, `nvcc` on
`PATH`, the `nvidia-cuda-nvcc` package, and standard install locations. ROCm
checks `ROCM_PATH`/`ROCM_HOME`, `hipcc` on `PATH`, and the standard ROCm
install location. Missing locations become an empty string.

## Import-Time Behavior

Normal `import tilelang` initializes logging, configures library paths,
imports TVM/native dependencies, loads the TileLang native library unless
`SKIP_LOADING_TILELANG_SO=1`, and exposes JIT, profiler, language, autotune,
lowering, layout, backend, math, and tool APIs.

Set environment variables that affect import or library discovery before
`import tilelang`. `TL_LOWER_TRACE` is also read at import time; setting it
after import does not enable the hook for that process.
