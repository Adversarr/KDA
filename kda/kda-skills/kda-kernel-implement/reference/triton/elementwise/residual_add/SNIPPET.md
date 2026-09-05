# Residual add (streaming skeleton)

Files: `kernel.py`, `reference.py`.

## Math

```
h = x + residual        fp32 add, stored as out_dtype (default x.dtype)
dx = dh, dresidual = dh (cast to each input's dtype)
```

The backward needs no kernel; autograd's identity is the fused adjoint.

## What to copy from it

- **Flat 1-D tiling**: `start = program_id * BLOCK` as int64, `offs = start + arange(BLOCK)`,
  `mask = offs < n`. This is the tail-safe skeleton for any elementwise stage inside a fusion.
- **Vector width**: `BLOCK = 2048` with 4 warps gives 16 elements per lane, i.e. 16-byte
  loads for bf16; `torch.add` runs at the same speed, and so does this kernel.
- **fp32 residual stream**: pre-norm blocks that keep the residual in fp32 pass
  `out_dtype=torch.float32`; the kernel stores one dtype while reading another, which is
  exactly what a residual + rmsnorm fusion does with its `h` output (compose this snippet with
  `fp32_norms/rmsnorm`, "Fusing with neighbours"; there is no separate fused snippet).

## Dtype and layout rules

Contiguous inputs of equal shape (the wrapper asserts). Mixed input dtypes are fine: both
are upcast to fp32 in registers.

## Fusing with neighbours

A residual add is never worth its own launch inside a fused block; it appears as the first
lines of a norm kernel (`h = x + r`, store `h`, then normalise `h`) or the last lines of an
epilogue. Its gradient contribution is a pass-through, which is why the fused backward of
`residual + norm` is just the norm backward with `dh` routed to both inputs.

## Gating

Gated variants (`h = residual + g * x`, `y = g * x` with `g` in (0, 1)) live under this
family: same skeleton, one more load. The gate gradient `dg = dh * x` reuses the loaded `x`
tile; keep `g` fp32 if the user keeps it fp32.
