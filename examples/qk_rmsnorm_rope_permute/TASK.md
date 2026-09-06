In my attention block the q and k projections go through RMSNorm over the head dim, then rotary
embedding, then a transpose to `(B, H, S, D)` for the attention kernel. Three kernels and a copy
each, for q and for k:

```python
def qk_prep(q, k, q_norm_w, k_norm_w, cos, sin, eps):
    # q, k: (B, S, H, D) bf16;  cos, sin: (S, D/2) fp32;  *_norm_w: (D,) fp32
    q = rmsnorm(q, q_norm_w, eps)
    k = rmsnorm(k, k_norm_w, eps)
    q = apply_rope(q, cos, sin)      # half pairing: (x1, x2) = (x[..., :D/2], x[..., D/2:])
    k = apply_rope(k, cos, sin)
    return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()
```

Shapes: B 8, S 2048, H 32, D 128, bf16 activations, fp32 norm weights and tables. Forward and
fused backward (dq, dk, dq_norm_w, dk_norm_w). One kernel launch for q and one for k is fine;
one launch for both is better.

The ordinary eager repository is in `user_repo/`; the definition to optimize is
`decoder/attention.py` and configuration is `decoder/config.py`. Run `python -m decoder --steps
50 --seed 0` for representative training or `python -m decoder --smoke --steps 3 --seed 0` for a
bounded correctness smoke. The latter preserves the operation's channel widths and head
dimensions while reducing batch/token counts. It prints `median_step_ms` and `final_loss`.

Please implement and verify forward and backward, benchmark the representative shapes, and
integrate the result at that definition site with an eager fallback.
