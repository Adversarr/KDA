# Advanced Environment And Import Behavior

This page is for debugging import, cache, target/backend defaults, and
development environment behavior.

## Dynamic Environment Reads

Most `tilelang.env` fields are backed by descriptors that read from
`os.environ` dynamically:

```python
import os
import tilelang

os.environ["TILELANG_VERBOSE"] = "1"
assert tilelang.env.get_default_verbose()
```

Assigning to a field such as `tilelang.env.TILELANG_DISABLE_CACHE = "1"` stores
a forced runtime override inside the descriptor. It does not automatically
write that value back into `os.environ`.

Prefer setting `os.environ[...]` before importing TileLang for import-time path
controls. Runtime assignment is mainly for tests and focused debugging.

## Light Import Mode

`tilelang.env.is_light_import()` is true when running under AutoDD's module
entry point. In light import mode, the top-level package skips heavy imports and
native library loading. This lets `python -m tilelang.autodd ...` start without
pulling in the full compiler stack before it begins reducing a target script.

Normal users should not rely on light import as a public partial-import mode.
It exists to support tooling.

## Native Library Loading

Normal import loads the TileLang native library unless:

```text
SKIP_LOADING_TILELANG_SO=1
```

Skipping the native library can help diagnose Python import/path issues, but it
does not produce a usable compile/run environment for ordinary kernels.

On non-Windows platforms, TileLang temporarily adjusts dynamic loader flags to
include lazy symbol binding while loading native dependencies. On Windows, it
adds runtime DLL directories for TileLang, TVM-FFI, CUDA, and related packages
where available.

## TVM And Template Paths

Environment variables used during path setup include:

| Variable | Purpose |
| --- | --- |
| `TVM_IMPORT_PYTHON_PATH` | Python path for importing TVM. |
| `TVM_LIBRARY_PATH` | Native library path for TVM-related libraries. |
| `TL_CUTLASS_PATH` | CUTLASS include path. |
| `TL_COMPOSABLE_KERNEL_PATH` | Composable Kernel include path. |
| `TL_TEMPLATE_PATH` | TileLang template/source path used by lowering. |

Set these before importing TileLang when overriding a source or development
installation.

## Cache State Layers

Kernel cache enablement is the conjunction of two states:

1. Runtime `CacheState`, controlled by `tilelang.enable_cache()` and
   `tilelang.disable_cache()`.
2. Environment disable flag, controlled by `TILELANG_DISABLE_CACHE`.

`tilelang.enable_cache()` cannot override `TILELANG_DISABLE_CACHE=1`.

Autotune result caching has a separate flag,
`TILELANG_AUTO_TUNING_DISABLE_CACHE`, because tuning results and compiled
kernels are different cache artifacts.

## Cache Namespace And Lib Stamp

On-disk kernels live under
`$TILELANG_CACHE_DIR/<version>/<os-arch>/kernels/`. Staging writes go to a
sibling `staging/` directory and are renamed atomically.

`TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1` folds a SHA-256 hash of the native
TileLang / TVM libraries into the cache key. Leave it off for ordinary
installs; turn it on when iterating on C++ passes without bumping
`tilelang.__version__`.

There is no `TILELANG_CLEAR_CACHE` and no public `tilelang.clear_cache()`.
Bypass with `TILELANG_DISABLE_CACHE=1` or delete a specific namespaced
directory.

## Windows-Specific Notes

On Windows, TileLang:

- prepends runtime DLL search paths to `PATH`
- registers DLL directories through Python's secure DLL loader
- defines an `os.RTLD_LAZY` sentinel because Windows does not support POSIX
  lazy `dlopen`
- defaults `TVM_FFI_RELEASE_GIL_BY_DEFAULT=0` to avoid thread-safety issues in
  TVM-FFI during concurrent autotune work

These settings are import-time behavior. Set overrides before importing
TileLang.

## Target/Backend Debugging

When a kernel behaves differently across machines, record:

- `TILELANG_DEFAULT_TARGET` (string or JSON dict)
- `TILELANG_EXECUTION_BACKEND`
- `TILELANG_VERBOSE`
- explicit `target` and `execution_backend` arguments
- `pass_configs`
- whether kernel cache and autotune cache were enabled
- whether `TILELANG_KERNEL_CACHE_USE_LIB_STAMP` was on

Those values are part of the practical compilation contract and often explain
why two runs do not use the same lowered or runtime path.
