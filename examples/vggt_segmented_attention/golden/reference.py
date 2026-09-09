"""Independent attention math with bounded score memory for large sequences.

The eager and compiled large references share Python chunk loops. Compilation covers
only chunk math and its explicit adjoint, never thousands of loop iterations.
"""
import torch
import torch.nn.functional as F

COMPILE_DESCRIPTION = "dynamic forward/adjoint chunks; Python loops; fp32 dK/dV accumulation"


def _float_product(a, b):
    """Matrix products with low-precision inputs and an fp32 output/accumulator."""
    shape = (*a.shape[:-2], a.shape[-2], b.shape[-1])
    left = a.reshape(-1, a.shape[-2], a.shape[-1])
    right = b.reshape(-1, b.shape[-2], b.shape[-1])
    if a.dtype == torch.float32:
        return torch.bmm(left, right).reshape(shape)
    return torch.bmm(left, right, out_dtype=torch.float32).reshape(shape)


def _chunk_math(q, k, v, mask, queries):
    with torch.autocast("cuda", enabled=False):
        logits = _float_product(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))
        return (_float_product(torch.softmax(logits, -1).to(v.dtype), v)).to(q.dtype)


def _chunk(q, k, v, mask, start):
    queries = torch.arange(start, start + q.shape[-2], device=q.device)
    return _chunk_math(q, k, v, mask, queries)


def _chunk_adjoint(q, k, v, mask, queries, upstream, output):
    """Return dQ in storage dtype and dK/dV contributions in fp32."""
    with torch.autocast("cuda", enabled=False):
        scale = q.shape[-1] ** -0.5
        logits = (_float_product(q, k.transpose(-2, -1))) * scale
        prob = logits.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
        # Standard Flash backward: fp32 dP and row delta, low-precision
        # P/dS tensor-core operands, then fp32 matrix-product accumulation.
        dp = _float_product(upstream, v.transpose(-2, -1))
        upstream = upstream.float()
        delta = (output.float() * upstream).sum(-1, keepdim=True)
        ds = (prob * (dp - delta)).to(q.dtype)
        return (((_float_product(ds, k)) * scale).to(q.dtype),
                (_float_product(ds.transpose(-2, -1), q)) * scale,
                _float_product(prob.to(v.dtype).transpose(-2, -1), upstream.to(v.dtype)))



class _BoundedReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, chunk_fn, adjoint_fn):
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
        ctx.save_for_backward(q, k, v, out)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out = ctx.saved_tensors
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
                        do[b:b+1, h:h+1, start:start+128],
                        out[b:b+1, h:h+1, start:start+128])
                    dq[b:b+1, h:h+1, start:start+128] = a
                    dk[b:b+1, h:h+1] += c
                    dv[b:b+1, h:h+1] += d
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None



class _FullReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask):
        out = _chunk_math(q, k, v, mask, None)
        ctx.save_for_backward(q, k, v, mask, out)
        return out

    @staticmethod
    def backward(ctx, upstream):
        q, k, v, mask, out = ctx.saved_tensors
        dq, dk, dv = _chunk_adjoint(q, k, v, mask, None, upstream, out)
        return dq, dk.to(k.dtype), dv.to(v.dtype), None


def _mask(q, lengths, tokens_per_view):
    mask = torch.zeros((q.shape[0], q.shape[2]), device=q.device, dtype=torch.bool)
    for b, row in enumerate(lengths):
        for view, n in enumerate(row):
            mask[b, view*tokens_per_view:view*tokens_per_view+n] = True
        if not sum(row) and q.shape[2]:
            mask[b, 0] = True
    return mask


def segmented_attention(q, k, v, lengths, tokens_per_view, *, recompute=False):
    if q.shape[0] == 0 or q.shape[2] == 0:
        return q + k + v
    mask = _mask(q, lengths, tokens_per_view)
    if q.shape[2] > 8192:
        return _BoundedReference.apply(q, k, v, mask, _chunk_math, _chunk_adjoint)
    return _FullReference.apply(q, k, v, mask)


def compiled_reference():
    forward = torch.compile(_chunk_math, fullgraph=True, dynamic=True,
                            mode="max-autotune-no-cudagraphs")
    adjoint = torch.compile(_chunk_adjoint, fullgraph=True, dynamic=True,
                            mode="max-autotune-no-cudagraphs")

    def run(q, k, v, lengths, tokens_per_view, *, recompute=False):
        return _BoundedReference.apply(q, k, v, _mask(q, lengths, tokens_per_view), forward, adjoint)
    return run


def sdpa(q, k, v, lengths, tokens_per_view, *, recompute=False):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=_mask(q, lengths, tokens_per_view)[:, None, None, :])
