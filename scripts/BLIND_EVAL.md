# Blind evaluation runner

`blind_eval.py` uses Python's standard library and the Docker CLI. Python 3.7+
works on the host; GPU execution uses the Python in
the container. No `pip install` or Docker Python SDK is required.

## Quick start

```bash
cd /path/to/KDA
python3 scripts/blind_eval.py doctor --model MODEL_ID
python3 scripts/blind_eval.py smoke --model MODEL_ID --gpus 0
python3 scripts/blind_eval.py run --model MODEL_ID --gpus 0,1
```

Choose GPUs assigned to your work. The example IDs above are not a reservation:
other GPU services may be running. This runner never terminates those services.
It locks GPUs against other runs by the same Unix user and records GPU memory and
utilization, including outside interference. A GPU UUID identifies each assignment.

`run` first performs a short model/tool/GPU smoke, stored under
`../KDA-blind-runs/preflight/`. Only a successful smoke starts the example queue.
The default is all directories containing both `TASK.md` and `user_repo/`, sorted
by name. Specify a model available to your account; the default budget is two hours per example.
Use `--examples name1,name2`, `--timeout 30m`, or `--output-root /absolute/path`
to override these defaults. JSON config timeouts are numbers of seconds.

Runs are foreground processes. Keep the terminal connected or use your own tmux.
Ctrl-C or SIGTERM stops the queue and active containers, preserving all artifacts.
You can also operate from another terminal:

```bash
python3 scripts/blind_eval.py status RUN_ID --watch
python3 scripts/blind_eval.py logs RUN_ID --example fused_residual_rmsnorm --follow
python3 scripts/blind_eval.py stop RUN_ID
python3 scripts/blind_eval.py resume RUN_ID --gpus 0,1
python3 scripts/blind_eval.py retry RUN_ID --example fused_residual_rmsnorm --gpus 0
```

An absolute batch path can replace `RUN_ID`; otherwise it is resolved under
`--output-root`. Logs follow the latest attempt selected when the command starts.
Raw stderr is also available in that attempt's `stderr.log`.

`resume` runs only never-started examples. It does not resume an agent chat or
automatically retry stopped, failed, or timed-out work. `retry` creates a new
numbered attempt from the original frozen inputs. Previous attempts remain intact.
Snapshots are hash-checked on resume/retry. A changed snapshot requires a new batch.

## Agent configuration

Choose `--agent cursor|pi|claude|codex`. All agents require `--model`;
Pi also requires `--provider`. Model IDs pass through unchanged. There is no
cross-agent model-name translation or automatic model substitution.

`--config FILE.json` accepts these keys, with explicitly supplied CLI flags taking
precedence:

| Key | Meaning |
|---|---|
| `agent`, `model` | Coding CLI and its exact model identifier |
| `provider` | Pi provider (other adapters reject this option) |
| `effort` | Pi thinking level, Claude effort, or Codex reasoning effort; Cursor uses its model ID |
| `image` | Local GPU image, default `kda:dev`; resolved to an immutable image ID |
| `runtime` | Optional Linux runtime directory mounted read-only at `/opt/agent` |
| `executable` | Executable inside the container, optionally under `/opt/agent` |
| `auth_file` | Host credential file, copied into a private temporary home |
| `auth_env` | Name of an existing environment variable containing credentials |
| `timeout` | Positive seconds per example, default 7200 |

The runtime must include its dependencies and work under the container's UID/GID.
Alternatively, provide an image containing the CLI and omit `runtime`. The runner
does not install or update coding agents. Cursor automatically discovers the full
version directory behind `~/.local/bin/cursor-agent` when available.

Example configurations (replace model IDs, paths, and provider as appropriate):

```json
{"agent":"cursor","model":"YOUR_MODEL_ID","image":"kda:dev"}
```

```json
{"agent":"pi","provider":"openai","model":"YOUR_MODEL_ID","image":"kda-pi:local","auth_env":"OPENAI_API_KEY","effort":"high"}
```

```json
{"agent":"claude","model":"YOUR_MODEL_ID","image":"kda-claude:local","auth_env":"ANTHROPIC_API_KEY","effort":"high"}
```

```json
{"agent":"codex","model":"YOUR_MODEL_ID","image":"kda-codex:local","auth_env":"OPENAI_API_KEY","effort":"high"}
```

Never put secret values in this JSON. The default credential-file locations are
Cursor `~/.config/cursor/auth.json`, Pi `~/.pi/agent/auth.json`, Claude
`~/.claude/.credentials.json`, and Codex `~/.codex/auth.json`. Only the selected
credential file is copied; the entire home/config/history is not inherited.
Temporary copies are writable for token refresh, then removed on cleanup. Refreshed
copies are not written back to host credentials. For long runs, use credentials
whose provider supports this lifecycle; refresh rejection fails visibly.

`doctor` checks the image, CLI version/options, credential configuration, GPU
inventory and container Python imports. It does **not** claim authenticated API
access. `smoke` sends a real request and checks the output of a small GPU operation,
single-device visibility, tool writing and the absence of the source checkout and
Docker socket. Unsupported CLI flags fail preflight; unavailable models/auth fail
the actual smoke. Unknown output events are retained, not treated as parser errors.

## Isolation and lifecycle

Each attempt gets a fresh container, workspace and home. Only that fixture,
original task and copied KDA skills are supplied; source checkout, golden kernels,
evaluation scripts, other attempts, host home and Docker socket are not mounted.
The KDA installer runs on the host in copy mode. Internal links are materialized
in each attempt; input links escaping a fixture are rejected.

Containers use the host UID/GID, `--init`, private IPC, 8 GiB shared memory and one
Docker-exposed GPU (`cuda:0` inside). Coding CLIs run with their noninteractive tool
permissions enabled inside this boundary. Network remains available for providers
and development dependencies; this is local answer/context isolation, not a
network anti-cheating sandbox.

Workers follow KDA sequentially in the same container. Pi extension discovery is
disabled; if fresh-context workers are unavailable, KDA must record `same_context`.
Cursor's configurable explore model and Claude's subagent model are set explicitly;
other worker choices follow the pinned main model and task instructions. These
settings are not proof of worker identity. Requested model, observed IDs/labels and
unconfirmed worker identities remain separate in state.

Stop first prevents new launches, then stops active containers with a 15-second
grace period. Exit reason/code is recorded before removal. An internal timeout
also ends the container if the host scheduler disappears. Orphans remain inspectable
until `status`, `stop` or `resume` obtains the batch lock and reconciles them. Such
attempts are marked `interrupted`, not silently accepted. Cleanup errors are retained
in state and retried on reconciliation. No global Docker prune is used.

## Output and interpretation

The default output is `KDA-blind-runs/` beside the source checkout. A batch contains:

- `manifest.json`, `doctor.json`, `snapshots/`: effective configuration, versions,
  source commit, fixture/skill hashes and frozen input copies.
- `state.json`, `gpu.jsonl`, `summary.json`, `summary.md`: lifecycle, last activity,
  GPU samples, requested/observed models and per-attempt results.
- `attempts/EXAMPLE/NUMBER/`: `workspace/`, `prompt.txt`, `trajectory.jsonl`,
  `stderr.log`, normalized `events.jsonl`, and any recovered logs.

State files belong to the host controller and are not mounted into agent containers.
Native logs and generated code persist after container deletion. Output may contain
task source code and model conversations; use appropriate filesystem access.

`completed` means the process exited successfully. `agent_verdict` comes only from
`kda_kernels/*/report.json` → `verification.verdict`, and is `unknown` if unavailable
or invalid. Agent reports are **not independent acceptance**. Version one does not
adapt the 14 withheld golden benchmarks to arbitrary generated packages.

## Validation

Run `doctor` and then `smoke` with your configured agent, model and GPU before
launching a batch. The smoke checks GPU execution, an optimizer update, tool
writes and workspace isolation. Repeat these checks after changing the image,
CLI version or credentials. A successful static check does not establish
end-to-end compatibility or numerical acceptance of a generated kernel.

Development test suites and deployment-specific validation records are not
included in this distribution. Use the local run artifacts to assess the
configuration you actually execute.

For macOS transfers, use Python `tarfile` or explicitly disable AppleDouble and
extended-attribute generation in the archiver. The runner excludes `._*`,
`.DS_Store`, and `__MACOSX` from fixtures and removes them from newly installed
skill snapshots. It does not delete arbitrary metadata files in the source tree.

The container also sets `USER`/`LOGNAME` and explicit TorchInductor/Triton cache
directories under its writable private home. Arbitrary host UIDs need not appear
in the image's `/etc/passwd`: PyTorch optimizer initialization otherwise can fail
inside `getpass.getuser()`. The model smoke now requires one successful CUDA AdamW
backward/update as well as the small tensor calculation, so import-only success
cannot hide this failure.
