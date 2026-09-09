"""Ordinary eager prefix-segment attention; no dense validity mask is needed."""

import torch


def _float_product(a, b):
    """Matrix products with low-precision inputs and an fp32 output/accumulator."""
    shape = (*a.shape[:-2], a.shape[-2], b.shape[-1])
    left = a.reshape(-1, a.shape[-2], a.shape[-1])
    right = b.reshape(-1, b.shape[-2], b.shape[-1])
    if a.dtype == torch.float32:
        return torch.bmm(left, right).reshape(shape)
    return torch.bmm(left, right, out_dtype=torch.float32).reshape(shape)


class _FlashPrecisionAttention(torch.autograd.Function):
    """Eager matrix math with FlashAttention's mixed-precision adjoint."""
    @staticmethod
    def forward(ctx, q, k, v, mask):
        with torch.autocast(q.device.type, enabled=False):
            scores = (_float_product(q, k.transpose(-2, -1))) * q.shape[-1]**-0.5
            if mask is not None:
                scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
            p = scores.softmax(-1)
            out = (_float_product(p.to(v.dtype), v)).to(v.dtype)
        ctx.save_for_backward(q, k, v, out)
        ctx.mask = mask
        return out

    @staticmethod
    def backward(ctx, upstream):
        q, k, v, out = ctx.saved_tensors
        with torch.autocast(q.device.type, enabled=False):
            scale = q.shape[-1]**-0.5
            do = upstream.float()
            scores = (_float_product(q, k.transpose(-2, -1))) * scale
            if ctx.mask is not None:
                scores = scores.masked_fill(~ctx.mask[:, None, None, :], float("-inf"))
            p = scores.softmax(-1)
            dp = _float_product(upstream, v.transpose(-2, -1))
            delta = (out.float() * do).sum(-1, keepdim=True)
            ds = (p * (dp - delta)).to(q.dtype)
            dq = ((_float_product(ds, k)) * scale).to(q.dtype)
            dk = ((_float_product(ds.transpose(-2, -1), q)) * scale).to(k.dtype)
            dv = (_float_product(p.to(v.dtype).transpose(-2, -1), upstream)).to(v.dtype)
        return dq, dk, dv, None


def segmented_attention(q, k, v, lengths, tokens_per_view):
    """Each CPU tuple row lists valid token-prefix lengths for one scene's views.

    Q remains full length, each scene attends across all its valid views, and an
    entirely empty scene uses original key zero as a dummy. No input is mutated.
    """
    outputs = []
    for batch, row in enumerate(lengths):
        parts = [
            slice(i * tokens_per_view, i * tokens_per_view + n)
            for i, n in enumerate(row)
            if n
        ]
        if not parts:
            parts = [slice(0, 1)]
        kp = torch.cat([k[batch : batch + 1, :, s] for s in parts], dim=2)
        vp = torch.cat([v[batch : batch + 1, :, s] for s in parts], dim=2)
        outputs.append(_FlashPrecisionAttention.apply(q[batch:batch+1], kp, vp, None))
    return torch.cat(outputs, dim=0)
