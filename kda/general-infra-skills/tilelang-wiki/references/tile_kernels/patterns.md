# TileKernels language patterns

Recurring TileLang forms in the copied DeepSeek TileKernels sources under
[`src/`](src/). Use with the house style in [README.md](README.md). MHC
samples below are **Manifold HyperConnection** ([`src/mhc/`](src/mhc/)),
not MLA. Tutorial MHC:
[`../../examples/deepseek_mhc/`](../../examples/deepseek_mhc/).

## `T.StridedTensor`

SF and non-contiguous activations are annotated with an explicit stride tuple.
Logical shape stays `(num_tokens, hidden)` / `get_sf_shape(...)`; the inner
stride is `1` and the outer stride is often `T.dynamic` so one compiled kernel
covers TMA-padded views.

```python
x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), in_config.dtype],
out_sf: T.StridedTensor[sf_shape, (out_sf_stride, 1), out_config.sf_dtype],
```

From
[`src/quant/per_token_cast_kernel.py`](src/quant/per_token_cast_kernel.py).
Same pattern on MoE expanded SF
([`src/moe/expand_to_fused_kernel.py`](src/moe/expand_to_fused_kernel.py))
and batched transpose
([`src/transpose/batched_transpose_kernel.py`](src/transpose/batched_transpose_kernel.py)).
Compact `T.Tensor` is used only when the buffer is known contiguous.

`token_stride` is a **factory** int (logical hidden, including FP4 packing).
`*_sf_stride` is `T.dynamic` because `alloc_scaling_factors` may return a
sliced TMA-aligned view.

## `T.alloc_reducer`

Cross-thread reductions that are not a simple `T.reduce_*` use a reducer,
then `T.finalize_reducer` before the value is consumed.

Stable top-k min-index
([`src/moe/topk_gate_kernel.py`](src/moe/topk_gate_kernel.py)):

```python
idx_reducer = T.alloc_reducer((1,), T.int32, 'min', replication='all')
T.fill(idx_reducer, T.max_value(T.int32))
# ... write candidates ...
T.finalize_reducer(idx_reducer)
```

Persistent MHC backward accumulates per-SM partials
([`src/mhc/head_compute_mix_kernel.py`](src/mhc/head_compute_mix_kernel.py)):

```python
mhc_scale_grad_reducer = T.alloc_reducer(1, T.float32, replication='all')
T.fill(mhc_scale_grad_reducer, 0)
for t in T.Persistent(...):
    mhc_scale_grad_reducer[0] += ...
T.finalize_reducer(mhc_scale_grad_reducer)
T.copy(mhc_scale_grad_reducer, mhc_scale_grad_partial[pid, :])
```

Also: SwiGLU clamp counters and `weight_grad`
([quant.md](quant.md));
[`src/mhc/post_kernel.py`](src/mhc/post_kernel.py),
[`src/mhc/pre_apply_mix_kernel.py`](src/mhc/pre_apply_mix_kernel.py),
[`src/mhc/pre_split_mixes_kernel.py`](src/mhc/pre_split_mixes_kernel.py),
[`src/mhc/norm_fn_kernel.py`](src/mhc/norm_fn_kernel.py).

## `T.Persistent` + occupancy

When the work is many more tiles than SMs, launch `T.Kernel(num_sms)` and
walk tiles with `T.Persistent`. `num_sms` is a compile-time factory argument
from `get_num_sms()`.

```python
with T.Kernel(num_sms) as pid:
    for t in T.Persistent(
        [T.ceildiv(num_tokens, token_block_size)],
        num_sms,
        pid,
        group_size=1,
    ):
        ...
```

From
[`src/mhc/head_compute_mix_kernel.py`](src/mhc/head_compute_mix_kernel.py)
and
[`src/mhc/pre_split_mixes_kernel.py`](src/mhc/pre_split_mixes_kernel.py).
`get_fused_mapping` is the other occupancy pattern: persistent **grid-sync**
(`T.Kernel(num_sms)` + `T.sync_grid()`), not `T.Persistent`.

## `get_num_sms()`

[`src/config.py`](src/config.py):

```python
def get_num_sms() -> int:
    if _num_sms == 0:
        return get_device_num_sms()  # torch.cuda.get_device_properties(...).multi_processor_count
    return _num_sms
```

`set_num_sms(n)` overrides for tests / occupancy experiments
(`0 < n <= device SM count`). Hosts pass the value into the factory so the
grid size is compiled in: `get_fused_mapping`, `group_count`, `aux_fi`,
engram gate, SwiGLU clamp-count path, MHC mix backward. Do not read SM count
inside the prim_func.

## `T.async_copy` + `T.ptx_wait_group`

Software-pipelined global→shared copies. Issue N async copies, then
`T.ptx_wait_group(k)` to wait until at most `k` groups remain outstanding
(`0` = wait for all).

Double-buffered MHC multilayer recompute
([`src/mhc/multilayer_recompute_kernel.py`](src/mhc/multilayer_recompute_kernel.py)):
prefetch the next layer's four tensors while computing the current; wait `4`
if a prefetch is in flight, else `0`.

```python
T.async_copy(next_layer_output_tensor[i_n, i0_h * h_blk], layer_output_shared[1 - phase, :])
T.async_copy(next_pre_mix_tensor[i_n, 0], pre_mix_shared[1 - phase, :])
T.async_copy(next_post_mix_tensor[i_n, 0], post_mix_shared[1 - phase, :])
T.async_copy(next_comb_mix_tensor[i_n, 0, 0], comb_mix_shared[1 - phase, :, :])
if i_layer + 1 < L_post:
    T.ptx_wait_group(4)
else:
    T.ptx_wait_group(0)
```

Engram gate
([`src/engram/engram_gate_kernel.py`](src/engram/engram_gate_kernel.py))
uses the same pair with `loop_layout=` on some copies and waits of `0`/`1`/`2`/`3`
matching the number of issued groups. This is explicit cp.async overlap, not
`T.Pipelined` + TMA.

## `T.make_tensor` pointer tables

When the kernel must walk a **runtime list** of same-shaped tensors (MHC
layers), the host packs `data_ptr()` values into a GPU `int64` table and the
kernel rehydrates each slot with `T.make_tensor`.

Host
([`src/mhc/multilayer_recompute_kernel.py`](src/mhc/multilayer_recompute_kernel.py)
`_make_ptr_tables_batched`): pin an `int64` CPU buffer, write each
`t.data_ptr()`, `to(device, non_blocking=True)`, split by list.

Kernel signature uses `T.Tensor[(L,), T.ptr]`, then:

```python
layer_output_tensor_0 = T.make_tensor(layer_output_ptrs[0], (n, h), T.bfloat16)
pre_mix_tensor_0 = T.make_tensor(pre_mix_ptrs[0], (n, mhc), T.float32)
T.async_copy(layer_output_tensor_0[i_n, i0_h * h_blk], layer_output_shared[0, :])
```

Layer count `L` / `L_post` is a factory specialization; token count `n` is
`T.dynamic`. Do not turn a Python `list[Tensor]` into a kernel argument.
