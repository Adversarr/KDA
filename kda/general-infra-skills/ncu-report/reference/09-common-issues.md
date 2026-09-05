# Common Issues & Gotchas

Collected solutions for the recurring frustrations of profiling CUDA kernels.

---

## ncu permissions

### `ERR_NVGPUCTRPERM: The user does not have permission to access NVIDIA GPU Performance Counters on the target device`

Two solutions:

**A) Use sudo (simplest on dedicated servers):**
```bash
sudo ncu [...]
```

**B) Make it persistent (preferred on shared servers):**
```bash
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/ncu.conf'
sudo update-initramfs -u
# reboot, then regular user can run ncu
```

### `Could not deploy stock section files to "/home/USER/Documents/NVIDIA Nsight Compute/..."`

Set `HOME` to a writable directory:
```bash
export HOME=/any/writable/path
ncu [...]
```

This warning is harmless but noisy. ncu falls back to reading from the CUDA install dir.

---

## `-k "regex:..."` matches nothing

1. **Start from the Python/kernel token, not from a guessed full name.** For DSL-generated kernels, the best first selector is usually `regex:.*some_kernel.*`.
2. **Confirm the target actually launched.** Run the Python profile driver without ncu first.
3. **Inspect `action.name()` after a first report.** The captured demangled name tells you how specific the token needs to be.
4. **Escape regex metacharacters if you target a more specific emitted name.**

### `-k "regex:..."` matches too much

This is common with DSL-generated names.

Fixes:

1. Narrow the token to the real kernel symbol rather than a generic wrapper name.
2. Use `-s` and `-c` if several matching launches happen in one run.
3. Separate warmup-only and steady-state launches in the profile driver if possible.

---

## Source view is empty / `action.source_info(pc)` returns None

The binary was compiled without `-lineinfo`. Add it to the nvcc invocation:
```bash
nvcc -O2 -std=c++17 -lineinfo -gencode=... kernel.cu -o harness
```

For JIT or framework-integrated builds:

- **TVM-FFI**: profile through a dedicated Python runner first, then fall back only if the runtime cannot expose useful source mapping.
- **PyTorch `torch.utils.cpp_extension.load`**: pass `extra_cuda_cflags=["-lineinfo"]` when possible and avoid debug-heavy builds.
- **CUTLASS / generated CUDA via build systems**: thread `-lineinfo` through the build flags if possible.
- **Triton**: source mapping can be limited; do the report-level analysis first and treat source-level attribution as best-effort.

---

## PM sampling returns nothing

1. **You didn't request it.** Add `--section PmSampling --section PmSampling_WarpStates` to the ncu invocation.
2. **vGPU / MIG environment.** PM sampling isn't supported under virtualization. Use metric-based (non-timeline) analysis only.
3. **Kernel too short.** Kernels under ~20 µs produce few PM samples; what comes back is dominated by warmup noise.
4. **Specific PM metric just isn't available on your GPU / driver / ncu combination.** Some `pmsampling:sm__throughput.*` or `pmsampling:dram__throughput.*` variants may return empty instance arrays even when other `pmsampling:smsp__warps_issue_stalled_*` series work fine on the same report. Always check `m.num_instances() > 0`; if the SM/DRAM timeline is empty, the stall-reason timelines are a reliable proxy.

---

## Correctness is out of scope here

If the kernel does not already pass correctness checks, stop and fix that before using this skill.

This skill is only for:

- selecting the right generated kernel to capture
- collecting Nsight Compute reports
- analyzing performance signals
- turning those signals into optimization directions

---

## First-run compile or warmup polluted the profile

Symptoms:

- the captured launch is much slower than every later launch
- the captured kernel name belongs to setup code instead of the real target
- the report mixes compile-time helper kernels with the steady-state kernel

Fixes:

1. Run the profile driver once outside NCU before collecting.
2. Add a `--warmup-only` mode or equivalent separation in the Python runner.
3. Use `-s` to skip the first matching launches.
4. Re-check the captured name from `action.name()` after collection.

---

## ncu takes forever to finish

1. **`--set full` needs 45+ replay passes.** That's normal — each pass reruns the kernel for a different metric group. If your kernel takes 3 ms, full profile takes ~15 s; 3 ms → 300 ms kernels are miserable. Mitigation: profile a smaller representative workload if possible.
2. **Kernel launches aren't isolated.** If your binary does other expensive work (data loading, CUDA context init) before every kernel launch, that runs on every replay too. Move it outside the profile window (ncu only profiles kernel launches matching `-k`).
3. **Don't use `-G` (debug).** It regresses performance ~100× and is useless for perf profiling.

---

## Kernel crashes / produces NaN only under ncu

1. **Profiler clock jitter.** Between replays, ncu resets GPU state. Kernels that depend on specific uninitialized values (bad practice) can behave differently. Fix: initialize all inputs.
2. **Out-of-order pre-replay memory**: ncu saves/restores GPU memory between replays but that can expose latent bugs like reading uninitialized global memory.

---

## Metric returns `None`

1. **Wrong metric name.** See [`08-metric-names.md`](08-metric-names.md). A name that is absent on this chip or ncu version returns `None` exactly like a name that was never collected — distinguish the two before treating either as a measurement.
2. **Metric not in the collected sections.** Add the relevant `--section` or `--set`.
3. **Value is legitimately missing.** Some metrics (like tensor pipe counters) return 0 rather than None when the feature wasn't used; but others return None for hardware not present.

Read through `Metrics` rather than a `try/except` that returns a default. A default hides
which of the three cases above you are in, and the middle one is the only one you can fix
by recapturing:

```python
from ncu_utils import Metrics

m = Metrics(action)
m.status("sm__throughput.avg.pct_of_peak_sustained_elapsed")  # absent | novalue | zero | value
m.resolve("sm_sol").text()   # tries the known aliases and reports which name it used
```

---

## `ncu_report` import fails

```bash
find /usr/local/cuda* -name "ncu_report*" -type f 2>/dev/null
# e.g. /usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python/ncu_report.py
export PYTHONPATH=$PYTHONPATH:/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python
python3 -c "import ncu_report; print('OK')"
```

If there's still an `ImportError`, check that the module is compatible with your Python version. The `_ncu_report*.so` compiled extension alongside `ncu_report.py` is built for one specific Python version.

---

## TVM-FFI specific

### "I can't profile my kernel, it's built by `tvm_ffi.cpp.build`"

- Locate the cached `.so` and `kernel.cu`:
  ```
  ~/.cache/flashinfer_bench/cache/tvm_ffi/<solution_hash>/
  ```
- You'll see `build.ninja` with the nvcc invocation — note it does *not* include `-lineinfo`.
- **Preferred path:** drive profiling from a dedicated Python runner first.
- **Alternative:** inject `-lineinfo` into `build.ninja` and recompile manually. But this breaks on the next TVM rebuild.

### "The Python benchmarking script runs but ncu sees no matching kernel"

`-k "regex:..."` must match the demangled name. Start from the Python-side token if one exists; only inspect the compiled object directly if the token is too ambiguous.

---

## PyTorch specific

### Profiling a PyTorch model's kernel

1. Identify the kernel. Use `torch.profiler` to name it, or look for the generated kernel from `torch.compile`.
2. The kernel is often auto-generated (Triton, CUTLASS, cuDNN). For Triton kernels specifically, names may vary across runs or shapes, so profile a single script invocation and confirm the captured name from the resulting report.
3. For `torch.compile`-generated Triton, inspect `TORCH_LOGS=+dynamo` or `TORCH_COMPILE_DEBUG=1` to see the emitted code.

---

## Dynamic names across shapes or autotune variants

Some DSL stacks emit names that change with:

- shape
- dtype
- autotune configuration
- dispatch path

Fixes:

1. Tag reports by specialization or workload, not just by operator name.
2. Keep the Python-level token stable and use the report's captured demangled name as evidence.
3. Do not assume one regex is correct for all specializations until you have checked it.

---


### Profiling CUDA Graph-captured kernels

ncu handles CUDA Graph launches fine — each captured kernel shows up as a separate "kernel launch". Use `-k` regex + `-c N` to target.

---

## Reproducibility

### Results jitter between runs

1. **Lock GPU clocks:**
   ```bash
   sudo nvidia-smi -lgc <boost_clock_mhz>    # check with nvidia-smi -q -d CLOCK
   # profile
   sudo nvidia-smi -rgc                      # unlock
   ```
2. **Enable persistent mode** (avoids driver unload between invocations):
   ```bash
   sudo nvidia-smi -pm 1
   ```
3. **Pin the CUDA stream** explicitly rather than relying on default stream.

### Reports don't match colleague's results

- Check ncu version (`ncu --version`). Metric names change between major versions.
- Check GPU driver version (`nvidia-smi`). Some metrics only exist on certain drivers.
- Check exact nvcc invocation — a stray `-G` or missing `-lineinfo` makes a big difference.

---

## Output interpretation

### "`sm__throughput = X%`, is that good?"

It depends on the kernel type:
- GEMM / matmul: should be 50%+ on B200. Below 30% is bad.
- Element-wise / reduction: usually 10-30%, because they're DRAM-BW-bound.
- Attention / recurrence kernels: varies wildly; compare against a reference implementation.

Always check Speed-of-Light alongside: `dram__bytes_read.sum.pct_of_peak_sustained_elapsed`. If DRAM is saturated, low SM throughput is expected and OK. If DRAM is idle AND SM is idle, you're latency-bound.

### "The details page says `Est. Speedup: X%` — is that reliable?"

Yes, mostly. NCU's rule engine does a reasonable job estimating individual rule impact. Caveats:

- The sum of all `Est. Speedup`s is usually > 100%, because rules overlap (fixing A might also help B) and one may be relative to a section while the next is relative to the whole kernel. Don't add them.
- Rules are per-pattern; the rule engine doesn't know which one is hardest/easiest to fix in your codebase.
- Use `Est. Speedup: X%` to rank patterns by magnitude; use your judgement for ease of implementation.
- Some rules give no estimate at all, Small Grid among them, and they are often the ones naming the real problem. Ranking by percentage and then cutting the list drops exactly those, which is why `analysis/summary_<tag>.txt` prints every rule.
