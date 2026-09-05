# KDA agent types

`kda-implementer` (M2 implement, M4 tune, M5 freeze) and `kda-verifier` (M3) as Markdown with
YAML front matter (pi reads `.agents/agents/*.md`; `.claude/agents/` and `.cursor/agents/` link
to them for Claude Code and the Cursor CLI, which reads both) and as TOML for Codex
(`.codex/agents/*.toml`). The bodies are the same text; edit both.

Front matter is read by three harnesses with different vocabularies, so it keeps to the
intersection:

- No `tools:` line. Built-in tool names differ (pi `read, bash, ...`; Claude Code `Read,
  Bash, ...`), pi warns on an unknown name and Claude Code refuses to spawn an agent whose list
  resolves to nothing. Omitted means all tools in both.
- The verifier has direct-edit deny lists in both spellings. Reports/audit and scratch
  can be written through the shell; source ownership is an instruction, not a filesystem
  boundary. Finalized `report.json.verification` is the M3 decision.
- `skills:` preloads the stage skill (pi and Claude Code); the Cursor CLI knows only `name`,
  `description`, `model`, `readonly`, `is_background` and ignores the rest, so the body says to
  read the SKILL.md when it is not preloaded. `allowed_subagents: none` is pi's nesting switch,
  the body repeats "do not spawn subagents" for the others.
- Codex: the implementer requests `sandbox_mode = "danger-full-access"` for GPU execution
  and compile caches outside the workspace; the verifier requests `workspace-write` for
  reports and scratch files. The host's permission policy controls effective access.
  The developer instructions are the Markdown body. Dispatch uses `fork_turns: "none"`,
  and both bodies tell the agent to ignore inherited conversation.
- Use sequential stages and consume completed results before advancing. The installed
  `kda-kernels/reference/agent-handoffs.md` describes capability-based dispatch. Without worker
  support, use serial roles and record `same_context` rather than requiring manual conversations.
