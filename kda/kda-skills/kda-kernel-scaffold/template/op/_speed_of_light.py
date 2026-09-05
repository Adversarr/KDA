"""Roofline model for `{{op}}`; must match the Roofline section of SPEC.md.

`estimate` receives the same keyword arguments as the functional API (tensors and params)
and returns the minimum bytes the phase must move and the FLOPs it must execute. Bytes count
every input read once and every output written once. Phases:

* ``fwd``: training forward; the aux tensors of SPEC ``saved_for_backward`` are written.
* ``infer``: forward under ``no_grad``; no aux bytes.
* ``bwd``: reads the saved tensors and every gradient output, writes every gradient input.
* ``bwd_recompute``: like ``bwd`` but the aux is recomputed from the inputs it depends on
  (read them instead of the aux) - only when SPEC ``recompute.available``.

Rules and worked examples: kda-kernel-implement/reference/common/speed-of-light.md.
"""

from typing import List, Tuple

import torch

# Compute unit doing the arithmetic: "fp32" (CUDA cores) or "bf16" (tensor cores, e.g. GEMM
# epilogues). A tensor-core ROOF also switches the harness to the *achievable* compute roof
# (cuBLAS on `gemm_shapes`, below) and to an autotuned torch.compile baseline.
ROOF = "fp32"
PHASES = ("fwd", "infer", "bwd", "bwd_recompute")


def estimate(phase: str, **inputs) -> Tuple[float, float]:
    """Return ``(bytes_moved, flops)`` for ``phase`` in ``PHASES``."""
    # TODO(scaffold): derive from SPEC.md. Example for a row-wise op y = f(x) * w with a per-row aux
    # and a weight gradient reduced from per-program fp32 partials (speed-of-light.md, backward rule):
    #   x = inputs["x"]; d = x.shape[-1]; n = x.numel(); rows = n // d; esize = x.element_size()
    #   sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    #   n_prog = min(-(-rows // max(1, 4096 // triton.next_power_of_2(d))), 2 * sms)  # reference geometry
    #   partials = 2 * n_prog * d * 4 + d * 4        # written by the kernel, read by the reduce, dw written
    #   fwd:   bytes = 2 * n * esize + rows * 4 (read x, write y, write aux);              flops = k * n
    #   infer: bytes = 2 * n * esize;                                                        flops = k * n
    #   bwd:   bytes = 3 * n * esize + rows * 4 + partials (read dy, x, aux; write dx, dw); flops = k' * n
    #   bwd_recompute: bytes = 3 * n * esize + partials (aux recomputed from x);            flops = (k + k') * n
    # `k`, `k'` are per-element op counts, so those factors only apply to row-wise ops. A GEMM
    # counts whole GEMMs instead (speed-of-light.md, GEMM section): with `gemm = 2 * M * N * K`
    # (ONE GEMM) fwd/infer = `gemm`, bwd = `2 * gemm` (dX = dZ W and dW = dZ^T X), bwd_recompute =
    # `3 * gemm` (the recomputed Z), i.e. 2/4/6 * M*N*K. `4 * gemm` for the backward doubles the
    # roof and shows up as sol_eff > 1. Bytes are each operand once per GEMM that touches it (dZ
    # is read by both backward GEMMs), plus the split-K partials `2*s*M*N*4` when the SPEC
    # geometry uses them. Bytes are `numel * element_size`, reported in decimal MB; keep SPEC
    # prose and this function term-for-term identical.
    raise NotImplementedError("{{op}} speed-of-light estimate is not written yet")


def gemm_shapes(phase: str, **inputs) -> List[Tuple[int, int, int]]:
    """The ``(M, N, K)`` GEMMs ``phase`` executes; the harness times cuBLAS on them as the roof.

    Only meaningful when ``ROOF`` is a tensor-core unit: the achievable compute roof is cuBLAS
    running these exact GEMMs (`_common/bench.py::matmul_ms`), the way the memory roof is a
    same-size copy. Return an empty list for a phase without GEMMs; leave the
    ``NotImplementedError`` for a CUDA-core op (the harness then uses the datasheet peak, which
    row-wise kernels never approach anyway because they are memory-bound).

    For ``Y = act(X W^T + b) + R`` with ``X: (M, K)``, ``W: (N, K)``:
      fwd / infer: ``[(M, N, K)]``
      bwd: ``[(M, K, N), (N, K, M)]``            dX = dZ W, dW = dZ^T X
      bwd_recompute: the forward GEMM plus the two above

    Attention (``flash_attention_2``): keep the ``NotImplementedError``. Its per-head GEMMs are
    skinny in ``K = head_dim`` and cuBLAS runs them slower than the fused kernel, so listing
    them reports ``sol_eff > 1``; the same-FLOP cube the harness falls back to is the fair
    achievable roof.
    """
    # TODO(scaffold): tensor-core ops only, e.g.
    #   m, k = inputs["x"].shape[-2], inputs["x"].shape[-1]; n = inputs["weight"].shape[0]
    #   fwd = [(m, n, k)]; bwd = [(m, k, n), (n, k, m)]
    #   return {"fwd": fwd, "infer": fwd, "bwd": bwd, "bwd_recompute": fwd + bwd}[phase]
    raise NotImplementedError("{{op}} has no tensor-core GEMMs (or gemm_shapes is not written yet)")


__all__ = ["ROOF", "PHASES", "estimate", "gemm_shapes"]
