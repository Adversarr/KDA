# STATUS: `{{op}}`

Owned by the orchestrator. Resume from this file and current evidence, not chat history.

- [ ] M0 scaffold and eager sanity complete
- [ ] M1 contract checked within authorized scope (or explicitly approved when requested)
- [ ] M2 both backend correctness checks and lint pass
- [ ] M3 complete finalized verification -> `report.json.verification`
- [ ] M4 tuning finished (pass, budget consumed, or correct best-so-far candidate)
- [ ] M5 measured configs frozen and confirmed by post-freeze M3
- [ ] M6 integration authorized, applied and recorded in E2E.md

| Field | Value |
|---|---|
| Created | {{date}} |
| Kernel backend | {{kernel_backend}} |
| Compute pattern | TODO from SPEC |
| Recompute | default off; record rationale and any user priority/configuration |
| Authorization | TODO source, scope, review-only/stage-review constraints, integration permission |
| Last dispatched run | 0 (every implement/tune/freeze dispatch, including repairs, increments; M3 shares its worker run) |
| M4 tune rounds used | 0 / 2 (reverted hypotheses count) |
| Implementation repair retries used | 0 / 2 |
| Freeze repair retries used | 0 / 2 |
| Iteration log | TODO run, mode, hypothesis, result; preserve consumed runs after restore |
| Verification context | independent / same_context |
| Validated GPUs | none |
| Last report | none |
| Candidate record | none (record iteration may precede Last dispatched run) |
| Checkpoint | none (operation-relative path) |
| Unresolved evidence | none |

## Harness patches

Record shared-runtime patches here with file, symptom and change before synchronization.
