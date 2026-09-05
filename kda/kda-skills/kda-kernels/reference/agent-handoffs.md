# Agent handoffs

Read this when dispatching a KDA stage. Use capabilities actually exposed by the host;
model choice does not imply a particular tool API. The host's instructions and permissions
apply to all workers. Never change model defaults merely to run KDA.

Prefer a separate implementer and verifier with fresh context. Provide the stage skill path,
repository root, operation directory, mode, iteration, GPU command, diagnosis/amendment and
candidate/checkpoint paths. Require the worker's completion report and inspect its files.
Workers perform one stage and do not spawn descendants. Independent stages are not run
concurrently on the same package or GPU measurement stream.

| Host capability | Dispatch |
|---|---|
| Codex collaboration tools | Use `kda-implementer` / `kda-verifier` when supported; otherwise give the stage explicitly. With `spawn_agent`, request `fork_turns: "none"`; wait for its completed result using the documented completion mechanism. A notification alone is not the report. |
| Claude Code or compatible Task/Agent tool | Select the installed agent type where supported, request foreground execution when the API exposes that option, and consume the result before advancing. |
| pi Agent tool | Use the installed type and its foreground mode; avoid a named/background worker when it would outlive the calling session. |
| Cursor or another worker API | Use installed types where recognized; otherwise include the SKILL path and complete dispatch explicitly. Verify the host's tool arguments rather than assuming another harness's spellings. |
| No worker/fresh-context capability | Perform serial role changes with the same file ownership, read list and verification steps. Set audit `independence: same_context`; do not claim independent review or ask the user to manage new conversations. If the user explicitly requires independence, report the missing capability. |

Source ownership is distinct from evidence writes. Implementers own backend code, helpers,
IMPL_NOTES and TUNED; verifiers audit source without editing it. Both may create their
stage-specific diagnostic artifacts and `_scratch/` inside the operation directory.
The orchestrator owns SPEC, harness/interface changes, STATUS, checkpoints and integration.

When profiling through `ncu-report`, dispatch a named kernel and its already-verified
workload/phase, the timing evidence and the exact question. Set the profile output root to
`<op_dir>/_scratch/profile/<unique-run>`. This KDA handoff uses isolated kernel evidence;
it does not claim whole-model critical-path attribution. Retain diagnostic numbers in the
stage's report before deleting scratch. Respect unavailable counters; never change host
permissions to obtain a measurement without authorization.
