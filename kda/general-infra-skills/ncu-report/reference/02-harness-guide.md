# Profile Target Guide

The preferred profiling target is a small standalone **Python profile driver** whose sole purpose is to import the kernel you want to profile, compile or JIT it if needed, construct representative inputs, and launch the steady-state kernel that Nsight Compute should capture.

This guide starts **after correctness has already been established**. Do not put correctness checks into the profiling driver and do not use this skill to validate numerical behavior.

---

## Preferred file shape

Use this file split whenever practical:

```text
some_kernel_impl.py
test_some_kernel_impl.py
profile_some_kernel_impl.py
debug_some_kernel_impl.py      # optional
```

For Nsight Compute, the important file is `profile_some_kernel_impl.py`. That runner should stay small, explicit, and free of unrelated application logic.

---

## What a good Python profile driver contains

1. The importable kernel module or wrapper function.
2. Any explicit compile or JIT step required to materialize the generated CUDA kernel.
3. Deterministic input creation appropriate for profiling.
4. An optional warmup launch if the first call performs compilation or cache population.
5. A single steady-state launch pattern that works with `ncu -c 1`.
6. A stable kernel token and recommended selector, typically `regex:.*<kernel_token>.*`.

Things that should **not** be in the profile driver:

- unrelated model code, training loops, or data loading pipelines
- repeated benchmark loops unless the runtime requires them to reach the target launch
- correctness sweeps across many shapes
- plotting or report generation

The job of the driver is to make one target kernel easy to capture reproducibly.

---

## Template

Use [`../helpers/profile_template.py`](../helpers/profile_template.py) as the preferred starting point.

Customize these areas:

1. **Kernel import boundary**: import the implementation module directly.
2. **Compile or JIT step**: build the generated kernel explicitly when possible.
3. **Input loading**: shape-matched synthetic or real workload tensors.
4. **Warmup logic**: only if the first run is not the steady-state kernel you want.
5. **Capture token**: the Python-level symbol name or another stable identifier that survives in the generated CUDA name.

---

## Picking the kernel token

For DSL-generated kernels, the most robust capture token is usually the Python-level kernel symbol name, for example `some_kernel`.

Recommended workflow:

1. start with the Python entrypoint or wrapper name
2. build the selector as `regex:.*some_kernel.*`
3. confirm the captured demangled name from the report via `action.name()`
4. refine the token only if the regex is too broad or too narrow

The token should be stable across runs for the same specialization family.

---

## Sanity check before profiling

Always run the profile driver once **without** ncu:

```bash
python3 profile_some_kernel_impl.py --shape ...
```

Expected behavior:

- the module imports cleanly
- compile or JIT succeeds
- any warmup completes
- the steady-state launch runs successfully
- the script prints the kernel token or recommended regex if implemented that way

If it crashes or hangs, fix that before adding ncu to the mix.
