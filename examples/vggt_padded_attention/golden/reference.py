"""Independent attention math with bounded score memory for large sequences.

The eager and compiled large references share Python chunk loops. Compilation covers
only chunk math and its explicit adjoint, never thousands of loop iterations.
"""
import torch
import torch.nn.functional as F

COMPILE_DESCRIPTION = "dynamic forward/adjoint chunks; Python loops; fp32 dK/dV accumulation"


def _chunk_math(q, k, v, mask, queries):
    with torch.autocast("cuda", enabled=False):
        logits = q.float() @ k.float().transpose(-2, -1) * (q.shape[-1] ** -0.5)
        logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))
        return (torch.softmax(logits, -1) @ v.float()).to(q.dtype)


def _chunk(q, k, v, mask, start):
    queries = torch.arange(start, start + q.shape[-2], device=q.device)
    return _chunk_math(q, k, v, mask, queries)


def _chunk_adjoint(q, k, v, mask, queries, upstream):
    """Return dQ in storage dtype and dK/dV contributions in fp32."""
    with torch.autocast("cuda", enabled=False):
        scale = q.shape[-1] ** -0.5
        qf, kf, vf = q.float(), k.float(), v.float()
        logits = (qf @ kf.transpose(-2, -1)) * scale
        prob = logits.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
        # The output cast's adjoint and every attention matmul are fp32 here.
        # Only dQ rounds per chunk; dK/dV round once after all query chunks.
        upstream = upstream.float()
        dp = upstream @ vf.transpose(-2, -1)
        ds = prob * (dp - (prob * dp).sum(-1, keepdim=True))
        ds = ds * scale
        return ((ds @ kf).to(q.dtype), ds.transpose(-2, -1) @ qf,
                prob.transpose(-2, -1) @ upstream)


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
            for h in range(0, q.shape[1], 1):
                for start in range(0, q.shape[2], 128):
                    out[b:b+1, h:h+1, start:start+128] = chunk_fn(
                        q[b:b+1, h:h+1, start:start+128], k[b:b+1, h:h+1],
                        v[b:b+1, h:h+1], mb, queries[start:start+128])
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
            for h in range(0, q.shape[1], 1):
                for start in range(0, q.shape[2], 128):
                    a, c, d = ctx.adjoint_fn(
                        q[b:b+1, h:h+1, start:start+128], k[b:b+1, h:h+1],
                        v[b:b+1, h:h+1], mb, queries[start:start+128],
                        do[b:b+1, h:h+1, start:start+128])
                    dq[b:b+1, h:h+1, start:start+128] = a
                    dk[b:b+1, h:h+1] += c
                    dv[b:b+1, h:h+1] += d
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None


def padded_attention(q, k, v, key_valid):
    if q.shape[2] > 8192:
        return _BoundedReference.apply(q, k, v, key_valid, _chunk_math, _chunk_adjoint)
    return _chunk(q, k, v, key_valid, 0)


def compiled_reference():
    """Build genuinely compiled chunk functions with bounded graph and score sizes.

    Called once per workload. Baseline validation warms both phases before timing;
    compiler failures propagate to the runner's ordinary baseline exclusion.
    """
    forward = torch.compile(_chunk_math, fullgraph=True, dynamic=True,
                            mode="max-autotune-no-cudagraphs")
    adjoint = torch.compile(_chunk_adjoint, fullgraph=True, dynamic=True,
                            mode="max-autotune-no-cudagraphs")

    def run(q, k, v, key_valid, *, recompute=False):
        return _BoundedReference.apply(q, k, v, key_valid, forward, adjoint)

    return run


def sdpa(q, k, v, key_valid):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=key_valid[:, None, None, :])
