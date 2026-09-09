# Compare blind outputs with goldens

`compare_goldens.py` runs **after** a blind batch. The comparison process can read
goldens; it must never be used as the blind agent's environment.

Inside the same CUDA image as the blind run, mount the checkout at `/repo`, the
batch at `/batch` (both read-only), and a writable output parent at `/out`:

```bash
python /repo/scripts/compare_goldens.py suite \
  --checkout /repo --run /batch --output /out/comparison --timeout 240
python /repo/scripts/compare_goldens.py summarize \
  --checkout /repo --output /out/comparison
```

Use Docker's explicit GPU selection, a dedicated container, `--init`, host UID/GID,
and a writable HOME/cache directory. Do not mount the batch at `/run`: NVIDIA's
container runtime needs to create its own files there. Use a tmux session for a
long run; `docker stop <comparison-container>` stops the whole comparison. The
script does not schedule GPUs or acquire the blind runner's GPU locks; select a
free GPU and do not run another comparison on it concurrently.

The suite uses each example's unchanged `benchmark_cases.py` factories. Each case
gets a fresh Python subprocess, deterministic inputs, and a shared random upstream
gradient for both implementations. It checks training outputs, inference outputs,
and gradients against the independent golden eager reference. Tolerances and
parameter-gradient reduction scaling come from KDA's existing verification runtime.
It also requires output/gradient dtypes to match. A failing implementation is not
timed. Generated packages are called with an explicit kernel backend, bypassing
integration fallbacks; this measures the kernel package, not the integrated model.

Public functions are discovered from the one generated `interface.py`; ambiguous
packages are errors. AdaLN's golden `silu` boolean maps to generated `act` enum.
No mathematical transformations, layout copies, or kernel changes are applied.

Timing uses the shared profiler helper: three warmups, ten iterations, three rounds
with alternating implementation order. Each round is a median; the reported value
is the minimum round median, matching the example benchmark convention. The raw
round values are retained. A missing profiler sample remains unavailable, without
switching timing methods. Forward, backward, and inference are separate. Ratios
are **agent milliseconds / golden milliseconds**: above one means agent is slower.

The initial suite excludes `real_*`, `model_*`, and recompute case variants and
records every exclusion. It includes available primary, small, tail, strided,
empty, and special-value cases; a case timeout remains a missing result rather
than a correctness failure. It is scoped comparison evidence, not full numerical,
performance, integration, or large-video acceptance. Goldens may fail too; their
presence does not make them an unquestionable numerical oracle.

Outputs include per-case JSON and stderr/stdout logs, a manifest with source
hashes, and a JSON/Markdown summary. Existing output directories are rejected.
`summarize` may run against a partial batch and records that it is incomplete.
Original goldens, blind outputs, and historical reports are never overwritten.
