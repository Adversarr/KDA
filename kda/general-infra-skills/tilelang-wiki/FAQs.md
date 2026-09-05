# Frequently Asked Questions

## [Question] `ParallelOpNode::RecordBufferAccess` fails on an interleaved complex tile

If you hit an error like:

```text
tvm.error.InternalError: Check failed: (StructuralEqual()(it->second.indices, indices)) is false: q_tile: (tid * 2 + 1,) and (tid * 2,)
```

or the 2D variant:

```text
tvm.error.InternalError: Check failed: (StructuralEqual()(it->second.indices, indices)) is false: q_tile: (tid, 1) and (tid, 0)
```

inside a `T.Parallel(...)` loop, a common cause is using one fragment buffer to
represent interleaved complex lanes and then reading sibling indices such as
real/imag from that same buffer in the parallel loop body.

This shows up naturally in RoPE-style kernels where the host data is laid out as
`[..., head_dim]`, but the kernel tries to treat it as complex pairs by loading
one tile and then accessing:

```python
q_real = q_tile[pair_idx, 0]
q_imag = q_tile[pair_idx, 1]
```

or:

```python
q_real = q_tile[real_idx]
q_imag = q_tile[imag_idx]
```

In current TileLang lowering, that access pattern can attach incompatible access
records to the same parallel buffer and fail during layout inference.

The robust workaround is to split the lanes into separate buffers before the
parallel loop:

```python
q_real_tile = T.alloc_fragment((tile_pairs,), dtype)
q_imag_tile = T.alloc_fragment((tile_pairs,), dtype)
freq_real_tile = T.alloc_fragment((tile_pairs,), freq_dtype)
freq_imag_tile = T.alloc_fragment((tile_pairs,), freq_dtype)

T.copy(q[..., 0], q_real_tile)
T.copy(q[..., 1], q_imag_tile)
T.copy(freqs[..., 0], freq_real_tile)
T.copy(freqs[..., 1], freq_imag_tile)
```

Then use one stable index per buffer inside `T.Parallel(...)`:

```python
for tid, lane in T.Parallel(threads, elements_per_thread):
    pair_idx = tid * elements_per_thread + lane
    q_real = q_real_tile[pair_idx]
    q_imag = q_imag_tile[pair_idx]
```

Rule of thumb: for complex-number or pairwise kernels, do not rely on
interleaved real/imag fragment accesses inside `T.Parallel(...)`. Either split
the lanes into separate buffers or move the pair extraction outside the parallel
region.

## [Question] A FullRow GEMM accumulator stores interleaved gate/up pairs, and the postprocess has to split them out manually

Another manifestation of the same limitation appears in GEMM-driven pairwise
epilogues such as RMSNorm + SwiGLU.

Suppose a GEMM writes one fragment tile `acc` whose columns are interleaved as:

```text
[gate_0, up_0, gate_1, up_1, ...]
```

The natural postprocess is to read sibling lanes from the same accumulator,
for example:

```python
gate = acc[i, j * 2]
up = acc[i, j * 2 + 1]
```

In current TileLang lowering, that sibling-fragment access pattern is the same
known issue as the minimal repro above. Even when the larger kernel is
restructured to extract `gate_frag` and `up_frag` in separate `T.Parallel(...)`
loops before applying RMSNorm and SwiGLU, that split is still a workaround for
the underlying layout-inference restriction.

This is why a larger fused kernel may end up written in the more awkward form:

```python
gate_frag = T.alloc_fragment((block_M, paired_outputs_per_tile), accum_dtype)
up_frag = T.alloc_fragment((block_M, paired_outputs_per_tile), accum_dtype)

for i, j in T.Parallel(block_M, paired_outputs_per_tile):
    gate_frag[i, j] = acc[i, j * 2]
for i, j in T.Parallel(block_M, paired_outputs_per_tile):
    up_frag[i, j] = acc[i, j * 2 + 1]
```

rather than reading both sibling lanes directly in one pairwise epilogue loop.

By contrast, Triton does not suffer from this specific layout-inference failure,
so the analogous fused GEMM + RMSNorm + SwiGLU implementation can keep the
pairwise logic in a more natural form, either by using separate accumulators for
gate and up or by reshaping/selecting lanes from an interleaved accumulator.

Practical workarounds in TileLang are:

1. Split the sibling lanes into separate buffers before the main pairwise
   `T.Parallel(...)` postprocess.
2. Materialize the interleaved result through shared memory before re-reading
   pair members.
3. Change the computation so gate and up are produced by separate GEMM paths or
   separate weight layouts instead of one interleaved fragment tile.

Rule of thumb: if a fragment produced by `T.gemm(...)` represents logical pairs
such as gate/up, real/imag, or even/odd lanes, treat direct sibling reads from
that fragment inside `T.Parallel(...)` as fragile. Split the lanes first, or
store them in a layout that avoids pair extraction from one fragment buffer.

## [Question] Layout infer conflict

If you hit an error like:

```text
tvm.error.InternalError: Layout infer conflict between acc_s and acc_s_cast in T.Parallel loop
```

the usual cause is that two `T.gemm(...)` calls expect different layouts for the
same intermediate buffer. In the report from issue
`tile-ai/tilelang#1165`, the first GEMM writes `acc_s` with
`policy=T.GemmWarpPolicy.FullCol`, while the later GEMM path expects a layout
compatible with `FullRow`.

Two known fixes are:

1. Change the casted buffer from fragment memory to shared memory:

```python
acc_s_cast = T.alloc_shared([block_N, block_M * heads], dtype)
```

This works because shared memory layout is more flexible.

2. Keep the casted buffer as a fragment, but align the GEMM layout policy:

```python
T.gemm(
    K_shared,
    Q_shared,
    acc_s,
    transpose_B=True,
    policy=T.GemmWarpPolicy.FullRow,
)
```

This is the preferred fix when valid for your kernel, because it avoids an
extra register-to-shared-memory copy and is typically faster.

Rule of thumb: if a fragment buffer is produced by one `T.gemm(...)` and later
consumed by another GEMM-related path, make sure both operations agree on the
fragment layout policy. If they do not, either align the policies or move the
intermediate through shared memory.

## [Question] `no available layout found` with two reductions

If you hit an error like:

```text
tvm.error.InternalError: Check failed: (min_reg_num < INT64_MAX) is false: no available layout found
```

and your kernel applies multiple reductions to the same fragment buffer, the
cause may be conflicting layout constraints from the reductions themselves. In
issue `tile-ai/tilelang#1714`, the kernel does:

```python
b = T.alloc_fragment([tilesize, nstr], dtype=dtype)
T.reduce_sum(R, b, dim=-1)
T.reduce_sum(R, b, dim=-2)
```

The problem is that `T.reduce_sum(...)` constrains both source and destination
layouts. Reducing into the same fragment buffer `b` with different reduction
dimensions can attach incompatible layout requirements, and layout inference
fails with `no available layout found`.

A simple workaround is to allocate the destination as shared memory instead of
a fragment:

```python
b = T.alloc_shared([tilesize, nstr], dtype=dtype)
```

Rule of thumb: if multiple reductions write into the same intermediate and the
reduction dimensions differ, avoid reusing a fragment buffer for all of them.
Use shared memory for the intermediate, or split the computation so each
reduction gets a layout-compatible destination.

## [Question] Observed `blockDim` does not match `threads=...`

If you write a kernel like:

```python
with T.Kernel(T.ceildiv(seq_len, block_m), heads, batch, threads=128) as (bx, by, bz):
```

but Nsight Compute shows `blockDim=(256, 1, 1)`, the usual reason is that TMA
was enabled and the compiler inserted an extra producer warp group.

In issue `tile-ai/tilelang#1523`, the TileLang maintainers explained that when
the compiler detects TMA copy usage, it may launch an extra warp group to issue
those TMA operations. That means the runtime CUDA block size can be larger than
the `threads=` value you passed in `T.Kernel(...)`.

So in this situation, the extra threads do not necessarily mean your
`threads=128` argument was ignored. Instead, TileLang augmented the launch
configuration to support TMA.

Rule of thumb: if profilers show more threads than expected, check whether the
kernel or pass configuration allowed TMA. When TMA is active, TileLang may add
producer warps on top of your requested worker threads.

## [Question] Autotune fails for every config on a metadata-driven kernel

If autotuning logs repeated validation failures or even reports that no
configuration succeeded, and your kernel takes structured metadata tensors such
as offsets, lengths, masks, or grouped-GEMM size tables, the usual cause is
that autotune is benchmarking the kernel with auto-generated inputs that do not
respect the metadata contract.

For example, a grouped kernel may expect inputs like:

```python
packed_lhs: T.Tensor((group_size, padded_M, padded_K), dtype)
packed_rhs: T.Tensor((group_size, padded_K, padded_N), dtype)
group_sizes: T.Tensor((group_size, 3), "int32")
```

where `group_sizes[g] = (M_g, N_g, K_g)` drives which output rows and columns
are valid for each group. If autotune generates arbitrary tensors for the data
inputs and metadata input independently, the reference program and the kernel
can disagree on what the valid region is, so every candidate config appears
wrong even when the kernel is fine.

The fix is to capture real, mutually consistent inputs with
`set_autotune_inputs(...)`:

```python
from tilelang.autotuner import AutoTuner, set_autotune_inputs

with set_autotune_inputs(packed_lhs, packed_rhs, group_sizes):
    result = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(out_idx=[-1], target="auto")
        .set_profile_args(ref_prog=packed_reference, skip_check=False)
        .run(warmup=3, rep=20)
    )
```

Two extra checks help a lot:

1. Ensure the reference program accepts the same input signature as the kernel's
   non-output inputs.
2. If you compare full packed outputs, define the padded or invalid region
   explicitly, for example by zero-filling skipped tiles, rather than leaving
   that region unspecified.

Rule of thumb: whenever kernel correctness depends on metadata tensors rather
than just shapes and dtypes, autotune with real captured inputs instead of
relying on automatic input generation.

## [Question] I changed the kernel source, but rerunning the profiler did not recompile it

If a profiling or autotuning rerun finishes suspiciously quickly, reuses an old
best config, or does not print the usual compile logs after you edited the
kernel, the usual cause is a cache hit rather than TileLang ignoring the edit.

TileLang has two on-disk layers under the same namespace:

```text
$TILELANG_CACHE_DIR/<version>/<os-arch>/kernels/
$TILELANG_CACHE_DIR/<version>/<os-arch>/autotuner/
```

The default root is `~/.tilelang/cache`. Example:
`~/.tilelang/cache/0.1.13/darwin-arm64/kernels/...`.

The **JIT kernel key is the lowered TIR**, serialized as
`func.script(show_meta=True)`. Python comments, unused helpers, and other
scaffolding around the same lowered function do **not** change the key.
Changing the kernel body, target, backend, pass configs, compile flags, or
output indices should.

Native-library content hashing is **opt-in**:

```text
TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1
```

When set, a SHA-256 stamp of `libtilelang` / TVM runtime libraries is folded
into the JIT key so C++ pass changes invalidate cache without a version bump.
Leave it off for ordinary installs.

Autotune persist is a separate disk cache. Passing `ref_prog`, `supply_prog`,
or `manual_check_prog` **disables autotune persist** for that tuner: callbacks
have no stable identity, so the tuner does not write or reuse a disk entry.

Ways to force a fresh run:

1. Set `TILELANG_AUTO_TUNING_DISABLE_CACHE=1` to disable autotune disk cache.
2. Set `TILELANG_DISABLE_CACHE=1` to disable the JIT kernel cache globally.
3. Delete the relevant namespaced directory under `$TILELANG_CACHE_DIR`.
4. Run the direct JIT path once without autotuning to confirm the new kernel
   itself recompiles.

For “which pass broke this?” after a forced recompile, use `TL_LOWER_TRACE`
and the checklist in `references/debug.md`. Do not start with
`TILELANG_PASS_DIFF` or `TL_ENABLE_DUMP_IR`.

Rule of thumb: if a kernel edit does not appear to trigger recompilation,
check whether the **lowered TIR** actually changed, then disable both cache
layers before assuming the compiler ignored the edit.

## [Question] Rank-1 partial-tile `T.copy` crashes in TMA layout inference

If a shared-to-global tail copy of a **rank-1** buffer fails during compile
with an internal `IndexError` inside bulk-store layout inference (accessing
`shape[dim - 2]`), that is issue `tile-ai/tilelang#2529`.

A partial last tile such as `T.copy(s_shared[0:remain], C[offset:offset+remain])`
used to fall through to the multidimensional `kBulkStore` path. That path
assumes a rank ≥ 2 shared source and crashes on a 1-D tail.

Current expected behavior: rank-1 shared sources skip the swizzled
multidimensional bulk-store path and use the linear / fallback copy path.
A partial 1-D tail should compile and match the `disable_tma=True` control.

1-D TMA still cannot represent out-of-bounds the way a 2-D descriptor can.
If you force `prefer_instruction="tma"` on a rank-1 region that is not
provably in-bounds and contiguous, lowering should refuse or fall back
rather than emit an unsafe 1-D TMA. Pad to a full tile, prove the region
in-bounds, or keep `disable_tma=True` on the tail copy.

Rule of thumb: a rank-1 partial tail is a linear copy, not a 2-D TMA store.
If compile still dies in `InferBulkLayout` on a 1-D source, treat it as a
regression of `#2529`.

## [Question] Unknown-sign `T.Ramp` fails CUDA lowering

If a vector index such as `A[T.Ramp(t - 2, 1, 4)]` fails on CUDA during
`LegalizeNegativeIndex` / `LegalizeSafeMemoryAccess` — often with an
assertion about a non-scalar bounds predicate — that is issue
`tile-ai/tilelang#2554`.

When the compiler cannot prove that each ramp lane is `< 0` or `>= 0`,
older lowering left the index unresolved. The safe-memory pass then
produced a vector-valued predicate, which later consumers reject.

Current expected behavior on CUDA: unknown-sign ramp lanes wrap like
Python negative indices, `Select(lane < 0, extent + lane, lane)`. For
`T.Ramp(t - 2, 1, 4)` over a length-1024 buffer this is per-lane wrap
such as `Select(t < 2, t + 1022, t - 2)`, not a compile failure.

```python
for t in T.serial(4):
    B[t, T.Ramp(0, 1, 4)] = A[T.Ramp(t - 2, 1, 4)]
```

At `t == 0` the load reads `A[1022], A[1023], A[0], A[1]`.

Rule of thumb: a runtime-dependent `T.Ramp` start is legal on CUDA and
wraps; do not rewrite it to scalar loops just to avoid the old assertion.

## [Question] `T.atomic_load` with a loop-carried index should compile

`T.atomic_load(status[look], memory_order="acquire")` is expected to
compile when `look` is a mutable scalar derived from a block index and
updated inside a `while` loop.

That pattern is issue `tile-ai/tilelang#2123`. `LowerAccessPtr` used to
assume `tl.access_ptr` argument 0 was still a `tir.BufferLoad` and raised:

```text
TypeError: Downcast from tir.Call to tir.BufferLoad failed.
```

Current expected behavior: the address-producing `BufferLoad` is preserved
through legalization, then lowered to `tir.tvm_access_ptr`. A kernel such
as:

```python
look = T.alloc_var(T.int32)
state = T.alloc_var(T.int32)
done = T.alloc_var(T.bool)
if tx == 0:
    look = tile - 1
    done = look < 0
    state = 0
while not done:
    state = T.atomic_load(status[look], memory_order="acquire")
    if state != 0:
        done = True
    else:
        look -= 1
        done = look < 0
```

should lower and run. If you still see the `Downcast` `TypeError`, treat
it as a regression of `#2123`.

Rule of thumb: `T.atomic_load` / `T.atomic_store` take a buffer element,
not a raw pointer. Dynamic or loop-carried indices are part of the
supported address form.

## [Question] ThreadSync errors in a non-warp-multiple divergent region

If compile fails with a fatal ThreadSync diagnostic that a required
shared-memory barrier sits inside a divergent region whose participating
thread count is not a multiple of the warp size (default 32), that is
issue `tile-ai/tilelang#2556`. This is a **new hard error**.

Previously `ThreadSync` could **silently drop** that barrier. A pattern
like:

```python
if tx < 48:
    S[tx] = T.float32(1)
    acc[0] += S[47 - tx]
```

needs a shared-memory sync between the write and the cross-thread read,
but `bar.sync` only accepts a warp-multiple participant count. Dropping
the sync left a data race; v0.1.13 refuses to compile instead.

Fixes:

1. Make the guarded range a warp multiple (`tx < 32`, `tx < 64`, …).
2. Hoist the shared write / cross-thread read so the required sync is
   outside the thread-divergent `if`.
3. Use a full `T.sync_threads()` at a block-uniform point when every
   thread must participate.

A 32-thread divergent region that needs a partial barrier still lowers.

Rule of thumb: do not put producer/consumer shared-memory pairing inside
`if tx < N` unless `N` is a multiple of 32. If you see the new fatal
error, change the guard or hoist the sync; do not look for a pass flag
that restores the old silent drop.

## [Question] Which pass broke this?

Prefer **`TL_LOWER_TRACE`** when asking which lowering pass changed the
IR or introduced a compile failure. Set it **before** `import tilelang`.

```bash
TL_LOWER_TRACE=1 python3 my_script.py
```

That writes a crash-safe HTML report plus per-pass `.tir` dumps under
`./tmp/lower_trace_dir` by default. Details and the short checklist are
in `references/debug.md`.

Do **not** start with the older hooks:

- `TILELANG_PASS_DIFF` is a legacy TVMScript line-diff. Leave it off
  unless you specifically want that text report
  (`TILELANG_PASS_DIFF_OUTPUT`).
- `TL_ENABLE_DUMP_IR` / `TL_DUMP_IR_DIR` dump IR at selected points; they
  do not tell you which pass first broke the module.

`TILELANG_PASS_PROFILE` / `TL_PASS_PROFILE_THRESHOLD_MS` time passes; they
do not show IR deltas.

Rule of thumb: reproduce with caches off
(`TILELANG_DISABLE_CACHE=1`, and `TILELANG_AUTO_TUNING_DISABLE_CACHE=1`
if tuning), then `TL_LOWER_TRACE=1`. Open `references/debug.md` before
adding `T.print` or AutoDD.
