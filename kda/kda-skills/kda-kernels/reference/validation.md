# Validation scope for SPEC v3 and runtime 0.7.0

Development validation covered 24 distinct regression tests on NVIDIA A800 and H20 with
torch 2.11.0+cu128, Triton 3.6.0 and nvmath-python 1.0.0. Each environment passed a 23-test
full run and a final 12-test runtime run including added legacy API coverage. Ten CPU-only
workflow tests also passed. This summarizes development checks, not production acceptance.

Covered behavior includes:

- Three capabilities across all three scaffold backends, read-only upgrade inspection and
  protection of existing implementations; legacy runtime calls and rejection of legacy evidence
  for new acceptance.
- Explicit verification backend selection, shared storage, nonzero offsets, gapped strides,
  output metadata, input mutation and partial or unused gradients.
- Auto fallback, strict backend selection, propagation of execution errors, and nvmath plan
  reuse, alignment, stream changes, operand release, eviction and exception cleanup.
- Measurement/cache compatibility, raw sample validation, balanced request A/B, failures and
  negative results, and distinct invalidation for contracts, executable code and ordinary prose.
- Generated forward-only, multi-output eager-adjoint, full-training and nvmath epilogue
  operations, including compile and report finalization.
- Two persistent worker processes loading actual candidate source, rank/backend/source
  witnesses, collective execution, rollback, success and exception cleanup, and acceptance
  through final manifest. A800 used one device with Gloo; H20 used two devices with NCCL.

The request fixtures each retained 13 invocations and explicitly accepted synthetic performance
limitations. Their results do not establish a production model speedup or independent production
approval. TileLang execution, other dependency versions, large-model quality and safe nvmath
graph capture remain unverified. Auto dispatch selects eager during unvalidated nvmath capture.

Development test suites, raw reports, traces, private source roots, hostnames and worker identity
records are not included in this public distribution. License notices and upstream attribution
are preserved. Runtime evidence still records the actual execution environment locally because
verification depends on it; review and redact a separate copy before publishing such evidence.
An anonymized summary does not replace the original evidence or certify a new deployment.
