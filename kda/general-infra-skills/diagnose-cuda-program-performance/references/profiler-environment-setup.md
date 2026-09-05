# Profiler environment setup

The goal is a trustworthy measurement, not a complete profiler installation. Reach section 6.1 with the Systems smoke passing, or report the exact preflight failure and continue: source reconnaissance in [cpu-mental-model-antipatterns.md](cpu-mental-model-antipatterns.md), degraded measurements in [profiling-and-attribution.md](profiling-and-attribution.md) section 4. Do not invent kernel internals, and do not stop. Report only claims the available tooling can support.

Flags and permission names here were current as of 2026-05. Verify a flag against `--help` before treating a mismatch as a skill bug.

## Contents

1. Establish which environment applies
2. Bare-metal prerequisites
3. Container launch contract
4. Image and host prerequisites
5. Preflight checklist
6. Minimal capture checks
7. Production capture hygiene
8. Troubleshooting
9. Environment receipt

## 1. Establish which environment applies

### 1.1 Recall an earlier verification

Check the repository's agent instructions for a `<cuda-perf-environment>` block first. If one exists, it holds the launch command, GPU selection, and permission state a previous investigation established, and most of this section collapses into running the preflight command it names. Confirm before trusting it: the GPU name and UUID still match, the container image is unchanged, and `nsys status --environment` still reports what the block claims. Where a check fails, that line is stale — re-verify it, Ask about anything it cannot answer, and update the block per [report-and-receipts.md](report-and-receipts.md) section 8.

### 1.2 Detect what the machine can tell you

Run these before asking anything:

```bash
test -f /.dockerenv && echo "container: docker"    # also check /run/.containerenv for Podman
grep -qa 'docker\|kubepods\|containerd' /proc/1/cgroup && echo "container: cgroup evidence"
echo "$KUBERNETES_SERVICE_HOST$SLURM_JOB_ID"        # non-empty implies a scheduler
command -v nsys ncu
nvidia-smi --query-gpu=index,uuid,name --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # other tenants
cat /proc/sys/kernel/perf_event_paranoid
```

Then route:

| Situation | Follow |
|---|---|
| Process runs directly on the host | 2, then 5 onward |
| Process runs in Docker, Podman, or a similar container | 3 and 4, then 5 onward |
| Process runs under Kubernetes, Slurm, or another scheduler | 3 and 4 for the container contract; the flags must be set in the pod or job spec, and counter permissions are a node property |
| Profilers cannot be installed or permitted at all | [profiling-and-attribution.md](profiling-and-attribution.md) section 4 |

Detection tells you where *this shell* runs. It does not tell you where the *workload* runs, and those differ whenever the shell is a dev container, a login node, or a laptop driving a remote host.

### 1.3 Detect/Ask detail under the SKILL.md moments

Ask moments live in [SKILL.md](../SKILL.md) section 4 — one batch per moment. This subsection is the Detect commands and permission reasons those moments need. It is not a second Ask batch.

| Section 4 question | Detect / permission reason |
|---|---|
| Where does the workload run, and can its launch command be changed? | 1.2 names this shell, not the workload. Section 3's two flags live in whoever's script or spec starts the container, which may not be reachable from here. Wrapper scripts routinely hide them. |
| Is the GPU shared, and may it be quieted? | `nvidia-smi --query-compute-apps` names other tenants. Contention invalidates timing; quieting a shared GPU is someone else's work. |
| May host settings (`perf_event_paranoid`, driver counters) or profiler installs be changed? | Both permissions are host policy (1.4 and 2). Changing them affects every user. Default: report the current value and the evidence it costs, then wait. |
| How long does one run take, and how many are acceptable? | Time one run first. Sets how many samples, and whether phase gating is mandatory rather than optional. |

State the answers back before a capture, per [SKILL.md](../SKILL.md) section 4. Where an answer does not arrive and work proceeds, **Flag** the assumption in the section 9 receipt and beside every number it affects. Once verified, offer to persist them so the next investigation Recalls instead of asking — [report-and-receipts.md](report-and-receipts.md) section 8.

### 1.4 Two permissions no container flag can grant

- **CPU sampling** needs `perf_event_open`, governed by the host's `perf_event_paranoid` setting and any seccomp policy.
- **Nsight Compute hardware counters** need profiling permission from the NVIDIA driver, governed by a host-level driver setting.

Nsight Systems tracing of CUDA APIs, kernels, copies, and NVTX generally works without either, which makes it the right first tool even in a restricted environment.

*Done when the environment class is chosen (bare metal, container, scheduler, or no-profiler), Detect output is recorded, and every section 4 gap is answered or Flagged.*

## 2. Bare-metal prerequisites

Verify: the NVIDIA driver is healthy; `nsys` is on `PATH` (`ncu` too if kernel analysis is already planned); the CUDA runtime/toolkit and framework match the driver; native code is built for the selected GPU architecture; the output directory is writable with ample free space; the selected GPU is idle enough for the requested confidence; CPU affinity, OpenMP, BLAS, and data-loader thread settings are the intended ones.

### CPU sampling permission

```bash
cat /proc/sys/kernel/perf_event_paranoid
```

Nsight Systems CPU sampling requires a value of `2` or lower; some features need lower still. This is the [SKILL.md](../SKILL.md) section 4 **Before touching the host** moment. Report the current value, say which evidence it costs, and wait. The setting is host-wide.

### Nsight Compute counter permission

Counter collection requires the driver to permit profiling for non-admin users, otherwise NCU reports `ERR_NVGPUCTRPERM`. Enabling it is a host-level change — an NVIDIA kernel module parameter, or running as an administrator — and cannot be granted from inside a container. Same host-touch moment: the change is host-wide, sometimes needs a driver reload that disrupts running jobs, and is a security decision on shared infrastructure. Report the error, name the setting, and wait.

If counters stay unavailable, Nsight Systems still supplies the whole-program timeline this skill depends on. Record the missing counters in the receipt instead of substituting guesses about kernel internals.

*Done when the checklist above is verified, `perf_event_paranoid` is recorded, and any host change is approved or explicitly declined.*

## 3. Container launch contract

When the workload is containerized, require both arguments in the effective launch invocation:

```text
--cap-add=SYS_ADMIN
-e NVIDIA_DRIVER_CAPABILITIES=all
```

Use an invocation shaped like:

```bash
docker run --rm --gpus 'device=GPU-UUID' \
  --cap-add=SYS_ADMIN \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /absolute/work:/work \
  -v /absolute/reports:/reports \
  IMAGE COMMAND
```

Prefer one selected physical GPU, ideally by UUID. Record how it maps to the container's logical device and to the application's `cuda:0` or equivalent.

Why these two:

- `--cap-add=SYS_ADMIN` enables `perf_event_open` under common Docker profiles so Nsight Systems can collect CPU sampling data. NVIDIA documents it as a supported narrower alternative to `--privileged`; host policy can still restrict access. See [Nsight Systems: Collecting Data Within a Container](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#collecting-data-within-a-container).
- `NVIDIA_DRIVER_CAPABILITIES=all` tells the NVIDIA Container Toolkit to mount all driver capability groups and their libraries and binaries, so tool visibility is not silently narrowed by defaults. See [NVIDIA Container Toolkit: Driver Capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities).

Use `SYS_ADMIN` rather than `--privileged`. If site policy forbids `SYS_ADMIN`, stop and coordinate with the administrator; tracing-only captures remain available in the meantime.

*Done when the effective launch — script, compose file, pod spec, or job spec — contains both flags, or the receipt records which one is missing and why.*

## 4. Image and host prerequisites

In addition to everything in section 2, verify: the NVIDIA Container Toolkit is configured for the runtime in use; the image contains a compatible `nsys` CLI (and `ncu` if kernel analysis is already planned), or approved host tools are mounted; profiling output goes to a writable bind mount with ample space; the CPU sampling policy in effect — `perf_event_paranoid`, seccomp profile, site restrictions — is known; the selected GPU is idle, checked from the host as well, since container visibility can hide competing jobs.

Execute a real kernel from every important native extension. A successful Python import does not prove architecture support.

*Done when the image, mounts, sampling policy, and host-side idle check are recorded.*

## 5. Preflight checklist

Run inside the exact environment that will perform the capture. This is the Systems preflight. NCU is not part of it.

### 5.1 Confirm GPU identity

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv
```

Require the intended visible device count and record the UUID. Check host-side processes too.

### 5.2 Confirm toolchain

```bash
nsys --version
nvcc --version
```

Also record framework and CUDA runtime versions from the application language.

### 5.3 Confirm the profiler environment

```bash
nsys status --environment
```

This reports whether sampling and tracing prerequisites are actually satisfied here. Treat a failure as a setup failure, not an application finding.

### 5.4 Confirm CUDA execution

Run a minimal framework CUDA operation followed by explicit completion; one kernel from each important native extension; and a correctness assertion on the result. If the runtime reports "no kernel image is available," rebuild for the architecture the GPU actually presents and verify the produced binary before profiling.

### 5.5 Confirm storage

Check free space and write then delete a small test artifact in the report directory. Full traces can reach hundreds of MiB.

*Done when 5.1–5.5 succeed in the capture environment, or the exact failing check is named and the investigation takes the [profiling-and-attribution.md](profiling-and-attribution.md) section 4 degraded path.*

## 6. Minimal capture checks

### 6.1 Nsight Systems smoke

Profile a tiny known CUDA workload:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=false \
  --output=/path/to/reports/nsys-smoke \
  COMMAND
```

Then verify: the report file exists and is nonempty; the expected process and kernel appear; CUDA API and device rows are present; NVTX appears if emitted; exit status and application correctness pass.

For CPU sampling, run a separate small capture with sampling enabled after `nsys status --environment` succeeds. Keep high-overhead sampling out of routine timing captures.

*Done when the Systems smoke report exists and those five checks pass.*

### 6.2 Nsight Compute smoke

Only after a named kernel is a proven target. Use a tiny, already-correct, stable kernel and a narrow launch filter, and confirm that at least one report action and metric come back. Start with that kernel, not a full application or an unbounded kernel set. NCU replay can generate several reports; re-check free space first. Insufficient-permission errors for hardware counters are a host policy matter; see section 2.

*Done when the filtered NCU report contains at least one action and metric, or the receipt records counters as denied.*

## 7. Production capture hygiene

- Expose exactly one intended GPU for timing authority where possible.
- Check the physical GPU for competing processes before starting, and again before retrying a run that needs to be quiet.
- Record clocks, power, and MIG state when relevant.
- Pin the CPU thread and library settings the workload requires.
- Use new output directories and `--force-overwrite=false`.
- Preserve historical traces; give every accepted report a unique basename.
- Gate capture to the relevant phase when the full workload is large.
- Keep clean timing, ownership tracing, CPU sampling, and NCU replay in separate captures.
- Verify application correctness and output identity after every capture.
- Check disk capacity before long captures and NCU replay.
- Record the complete effective launch command; wrapper scripts can hide missing flags.

*Done when the production capture uses a unique output path, a recorded launch command, and a quiet-GPU check taken immediately before the run.*

## 8. Troubleshooting

**`nsys status --environment` reports sampling disabled.** Likely causes: `perf_event_paranoid` above 2 on the host; a restrictive seccomp profile; missing `--cap-add=SYS_ADMIN` in a container; rootless runtime restrictions; site policy. Inspect the effective launch and the host setting, and coordinate with the administrator. Record that CPU-gap classification is unavailable and classify idle from [nsys-workflow.md](nsys-workflow.md) section 6.4 (`## No-CUDA time by host API in flight`), which does not need sampling.

**CUDA works but no kernels appear in Systems.** Check that the correct child process was traced; the capture range is actually entered; the profiler runs in the same container and namespace as the workload; the trace includes CUDA; the application did not exit before its asynchronous work completed; injected driver and tool libraries are visible; and process filtering and fork/exec behavior are as expected.

**NCU reports permission or counter errors.** Check the host driver counter policy first (section 2), then `SYS_ADMIN` presence in a container, a conflicting profiler or monitoring process, MIG restrictions, and tool/driver compatibility. Record missing metrics as unavailable.

**"No kernel image is available."** Rebuild every relevant extension for the target architecture and verify real kernel execution. Clear or isolate stale build caches if necessary, preserving user artifacts.

**The profiler sees the wrong GPU.** Compare the host UUID, `nvidia-smi -L` inside the environment, `CUDA_VISIBLE_DEVICES`, the framework's logical index, and the receipt. Numeric indices do not survive container remapping.

**Child-process activity is absent.** Inspect process-tree and fork tracing options, container PID namespace, launcher behavior, and capture-range placement. Profile the actual worker command directly when possible.

**Trace output is empty or disappears.** Use a writable absolute path, verify free space, and wait for profiler finalization. Leave report generation running until the artifact is written, unless the capture is being abandoned.

**The application times out under NCU.** Metric replay can execute a kernel many times. Isolate fewer launches, use a representative captured workload, filter by kernel and launch, and reduce sections. NCU is not an end-to-end profiler.

**Profiled time is much worse than clean time.** Reduce range cardinality, stack/shape/memory tracing, CPU sampling, backtraces, and capture span. Use the instrumented trace for ownership and counts, and return to a clean run for acceptance timing.

*Done when the matching row's action has been taken, or the failure is recorded in the receipt as a known restriction.*

## 9. Environment receipt

Record:

```text
Host:
Date/time/timezone:
Bare metal or container:
Container image/digest (if any):
Complete effective launch command:
SYS_ADMIN present (container): yes/no/not applicable
NVIDIA_DRIVER_CAPABILITIES (container): all/other/not applicable
perf_event_paranoid:
NCU counter permission: available/denied
Host GPU index/UUID/model:
Visible/logical GPU index:
Application device:
MIG/power/clocks:
Driver version:
CUDA runtime/toolkit:
Framework/compiler:
nsys version:
ncu version:
Native extension architectures:
CPU model/affinity:
OMP/BLAS/data-loader threads:
Profiler status output:
CUDA smoke result:
Systems smoke result:
Compute smoke result:
Artifact filesystem/free space:
Competing process check:
Known restrictions:
```

The environment is ready when the required profiler mode and the actual application CUDA path both succeed. When one of them does not, the receipt is what makes the resulting evidence interpretable — record it either way. The receipt belongs with the run. The subset that will be identical next time — the launch command, the GPU to use, the permission state, the preflight command — belongs in the project memory block, so this section becomes a confirmation rather than a discovery. See [report-and-receipts.md](report-and-receipts.md) section 8.

*Done when every field above is filled or marked not applicable, and Systems smoke plus the application CUDA path are both recorded.*
