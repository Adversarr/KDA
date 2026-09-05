# MoE routing recipes

Production routing lives in [`src/moe/`](src/moe/).
Public surface: `topk_gate`, `topk_sum_and_topk_group_idx`, `top2_sum_gate`,
`get_fused_mapping`, `expand_to_fused` / `expand_to_fused_with_sf`,
`reduce_fused`, plus helpers (`normalize_weight`, `group_count`,
`mask_indices_by_tp`, `inplace_unique_group_indices`, `aux_fi`).

Tutorial counterpart:
[`../../examples/topk/example_topk.py`](../../examples/topk/example_topk.py)
is an **unstable**, autotuned, eager top-k (returns values + `int32` indices,
zeros ties with `-10000`). Do not copy that contract into a TileKernels-style
router.

## Stable top-k

[`src/moe/topk_gate_kernel.py`](src/moe/topk_gate_kernel.py)
— `topk_gate(scores, num_topk) -> int64 [N, K]`.

- Contiguous `float32` scores `[N, E]`. Host allocates `int64` indices.
- Empty batch returns the empty tensor (no launch).
- One token per CTA, 32 threads. Pad `E` to a warp multiple with `-inf`.
- Repeat `K` times: `T.reduce_max` for the winning score, then
  `T.alloc_reducer(..., 'min')` over indices that match that score so **ties
  take the smaller expert id**. Knock the winner to `-inf` and continue.
- Output is contiguous. Docstring: "Always return the smaller index when there
  are ties."

```python
for k in T.unroll(num_topk):
    T.reduce_max(scores_fragment, amax_fragment)
    T.fill(idx_reducer, T.max_value(T.int32))
    for i in T.Parallel(num_aligned_experts):
        if scores_fragment[i] == amax_fragment[0]:
            idx_reducer[0] = T.min(idx_reducer[0], idx_fragment[i])
    T.finalize_reducer(idx_reducer)
    topk_idx_shared[k] = idx_reducer[0]
    for i in T.Parallel(num_aligned_experts):
        if idx_fragment[i] == idx_reducer[0]:
            scores_fragment[i] = -T.infinity(T.float32)
```

Grouped (DeepSeek-style) top-k uses
[`src/moe/common.py`](src/moe/common.py)
`get_topk_group_idx` +
[`src/moe/topk_sum_and_topk_group_idx_kernel.py`](src/moe/topk_sum_and_topk_group_idx_kernel.py):
per-group top-1 or top-2 sum, then a **stable** rank (`count` groups that are
strictly larger, or equal with a smaller group id). `num_groups <= 32`.
`top2_sum_gate` is the full grouped gate that also emits weights / mapped ids
([`src/moe/top2_sum_gate_kernel.py`](src/moe/top2_sum_gate_kernel.py),
scoring helpers in [`src/moe/scoring.py`](src/moe/scoring.py)).

## `get_fused_mapping`

[`src/moe/get_fused_mapping_kernel.py`](src/moe/get_fused_mapping_kernel.py)
builds the expert-major fused layout from `topk_idx: int64 [N, K]`.

Host args: `num_experts`, `num_expanded_tokens`, `alignment`,
`force_no_sync=False`. If `num_expanded_tokens == 0` and not `force_no_sync`,
the launcher over-allocates
`(N * K + (alignment - 1) * E) // alignment * alignment`, launches, then
`.tolist()` on `num_tokens_per_expert` and trims the pos buffers (this is a
**host sync**).

Grid is **one CTA per SM** (`get_num_sms()`), 256+ threads (doubled until
`>= num_experts`, cap 1024). Extra pass configs:
`TL_DISABLE_THREAD_STORAGE_SYNC`, `TL_DISABLE_OUT_OF_BOUND_WARNING`.

Returns:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `pos_to_expert` | `[E_exp]` | expert at fused slot; `-1` if pad |
| `pos_to_token` | `[E_exp]` | source token; `-1` if pad |
| `pos_to_token_topk` | `[E_exp]` | flat `(token * K + k)`; `-1` if pad |
| `token_topk_to_pos` | `[N, K]` | fused slot for that routing; `-1` if masked |
| `expert_start` / `expert_end` | `[E]` | aligned fused range |
| `num_tokens_per_expert` | `[E]` | aligned counts |
| `num_tokens_per_expert_list` | Python list | filled only on the host-sync path |

Algorithm, in order:

1. Zero / fill mappings with `-1`. Each warp atomics a local expert histogram
   into `experts_sum_per_warp_shared`.
2. Warp-prefix the histogram; write `num_experts_per_sm[sm, expert]`.
3. **`T.sync_grid()`** so every SM sees every SM's counts.
4. Each expert thread (tid `< E`) sums counts, aligns to `alignment`, exclusive
   `T.cumsum` for `expert_start` / `expert_end` (SM 0 writes globals).
5. Second pass scatters tokens into fused slots. Same-expert lanes in a warp
   rank themselves with PTX `__match_any_sync` + `T.popcount` so the prefix
   add is warp-local, not a global atomic:

```python
mask = T.call_extern(T.uint32, '__match_any_sync', 0xFFFFFFFF, expert_idx)
count = T.popcount(mask & lane_mask)
if i < numel and expert_idx >= 0:
    prefix_count = experts_sum_per_warp_shared[warp_idx, expert_idx]
    pos = prefix_count - count
    ...
    token_topk_to_pos_1d[i] = pos
    pos_to_expert[pos] = expert_idx
    pos_to_token[pos] = i // num_topk
```

Masked experts (`topk_idx == -1`) stay unmapped. Padded fused slots keep `-1`.

## Expand / reduce fused

[`src/moe/expand_to_fused_kernel.py`](src/moe/expand_to_fused_kernel.py)
— token-major `[N, H]` → expert-major `[E_exp, H]`.

- Grid `max(N, E_exp)`, 64 threads.
- Slots with `pos_to_expert < 0` are zeroed (activation and optional SF).
- Each token copies into every valid `token_topk_to_pos[k]`.
- `expand_to_fused_with_sf` also gathers SF. Packed UE8M0 **requires** TMA
  col-major SF. TMA SF is allocated `(H_sf, align(E_exp, 4))` then sliced /
  transposed like the quant ABI. `num_per_channels` in `{32, 128}`.

[`src/moe/reduce_fused_kernel.py`](src/moe/reduce_fused_kernel.py)
— inverse weighted sum. `hidden % 256 == 0`.

- One CTA per token, 128 threads. Unroll `K`; skip `pos < 0`.
- Optional `topk_weights`, optional per-slot `x_sf` (`QuantTensor` input),
  optional scalar `sf` when writing FP8 E4M3.
- Host may pass a preallocated `out`. Empty `N` skips the launch.

Typical pipeline: `topk_gate` (or grouped gate) → `get_fused_mapping` →
`expand_to_fused[_with_sf]` → expert GEMM → `reduce_fused`. Fused SwiGLU+cast
in [quant.md](quant.md) consumes `pos_to_token_topk` / `pos_to_expert` from
this mapping.

Helpers (same house style):
[`src/moe/group_count_kernel.py`](src/moe/group_count_kernel.py),
[`src/moe/normalize_weight_kernel.py`](src/moe/normalize_weight_kernel.py),
[`src/moe/mask_indices_by_tp_kernel.py`](src/moe/mask_indices_by_tp_kernel.py),
[`src/moe/inplace_unique_group_indices_kernel.py`](src/moe/inplace_unique_group_indices_kernel.py),
[`src/moe/aux_fi_kernel.py`](src/moe/aux_fi_kernel.py).
