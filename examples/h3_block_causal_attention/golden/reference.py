"""Independent attention math with bounded score memory for large sequences.

The eager and compiled large references share Python chunk loops. Compilation covers
only chunk math and its explicit adjoint, never thousands of loop iterations.
"""
import torch
import torch.nn.functional as F

COMPILE_DESCRIPTION = "dynamic forward/adjoint chunks; Python loops; fp32 dK/dV accumulation"


def _chunk_math(q, k, v, mask, queries):
    with torch.autocast("cuda", enabled=False):
        scores = (q @ k.transpose(-2, -1)).float() * (q.shape[-1] ** -0.5)
        keys = torch.arange(k.shape[-2], device=q.device)
        allowed = queries[:, None] // mask >= keys[None, :] // mask
        probs = scores.masked_fill(~allowed, float("-inf")).softmax(-1).to(v.dtype)
        return probs @ v


def _chunk(q, k, v, mask, start):
    queries = torch.arange(start, start + q.shape[-2], device=q.device)
    return _chunk_math(q, k, v, mask, queries)


def _chunk_adjoint(q, k, v, mask, queries, upstream):
    """Return dQ in storage dtype and dK/dV contributions in fp32."""
    with torch.autocast("cuda", enabled=False):
        scale = q.shape[-1] ** -0.5
        keys = torch.arange(k.shape[-2], device=q.device)
        allowed = queries[:, None] // mask >= keys[None, :] // mask
        score = (q @ k.transpose(-2, -1)).float() * scale
        prob = score.masked_fill(~allowed, float("-inf")).softmax(-1)
        prob_low = prob.to(v.dtype)
        # dP rounds at the bf16 probability/value matmul boundary. The scaled
        # score adjoint rounds again before the bf16 Q/K matmul adjoint.
        dp = (upstream @ v.transpose(-2, -1)).float()
        ds = prob * (dp - (prob * dp).sum(-1, keepdim=True))
        ds_low = (ds * scale).to(q.dtype)
        dq = ds_low @ k
        dk = ds_low.float().transpose(-2, -1) @ q.float()
        dv = prob_low.float().transpose(-2, -1) @ upstream.float()
        return dq, dk, dv


class _BoundedReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, chunk_fn, adjoint_fn):
        ctx.save_for_backward(q, k, v)
        ctx.mask = mask
        ctx.adjoint_fn = adjoint_fn
        out = torch.empty_like(q)
        queries = torch.arange(q.shape[2], device=q.device)
        # Tensor query offsets avoid specializing a graph for every Python start.
        for b in range(q.shape[0]):
            mb = mask[b:b+1] if isinstance(mask, torch.Tensor) else mask
            for h in range(0, q.shape[1], 4):
                for start in range(0, q.shape[2], 128):
                    out[b:b+1, h:h+4, start:start+128] = chunk_fn(
                        q[b:b+1, h:h+4, start:start+128], k[b:b+1, h:h+4],
                        v[b:b+1, h:h+4], mb, queries[start:start+128])
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        queries = torch.arange(q.shape[2], device=q.device)
        for b in range(q.shape[0]):
            mb = ctx.mask[b:b+1] if isinstance(ctx.mask, torch.Tensor) else ctx.mask
            for h in range(0, q.shape[1], 4):
                for start in range(0, q.shape[2], 128):
                    a, c, d = ctx.adjoint_fn(
                        q[b:b+1, h:h+4, start:start+128], k[b:b+1, h:h+4],
                        v[b:b+1, h:h+4], mb, queries[start:start+128],
                        do[b:b+1, h:h+4, start:start+128])
                    dq[b:b+1, h:h+4, start:start+128] = a
                    dk[b:b+1, h:h+4] += c
                    dv[b:b+1, h:h+4] += d
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None


def block_causal_attention(q, k, v, block_size):
    if q.shape[2] > 8192:
        return _BoundedReference.apply(q, k, v, block_size, _chunk_math, _chunk_adjoint)
    return _chunk(q, k, v, block_size, 0)


def compiled_reference():
    """Build genuinely compiled chunk functions with bounded graph and score sizes.

    Called once per workload. Baseline validation warms both phases before timing;
    compiler failures propagate to the runner's ordinary baseline exclusion.
    """
    forward = torch.compile(_chunk_math, fullgraph=True, dynamic=True,
                            mode="max-autotune-no-cudagraphs")
    adjoint = torch.compile(_chunk_adjoint, fullgraph=True, dynamic=True,
                            mode="max-autotune-no-cudagraphs")

    def run(q, k, v, block_size, *, recompute=False):
        return _BoundedReference.apply(q, k, v, block_size, forward, adjoint)

    return run


def sdpa(q, k, v, block_size):
    idx = torch.arange(q.shape[2], device=q.device)
    allowed = idx[:, None] // block_size >= idx[None, :] // block_size
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
