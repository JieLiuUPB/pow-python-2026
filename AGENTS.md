## Important rules

- Default to YAGNI, KISS, DRY, and prefer the shortest clear solution.
- Build modular first. Code files should preferably be ≤ 300 lines of code, unless forcing a split would sacrifice maintainability. Documentation, plans etc. can be as long as needed, but code files must be modular.
- Plan for known requirements, not hypothetical ones. Avoid speculative abstractions; keep stable boundaries where useful.
- If a task genuinely needs more code, split it rather than forcing a file under the LOC guideline.
- Do not add default fallbacks during development phase. If something fails, let it fail, so we can fix it!
- Do not leave empty try-catch blocks anywhere!
- Do not reinvent the wheel! Use open source, self-hosted libraries when needed. Ask the user, and help them qualify their selection.
- Design UI for the end-user, not for the schema!
- Before finishing, interrogate your diff: delete what is unnecessary, then simplify what remains. Prefer deleting over simplifying, simplifying over optimizing. If it is already good, leave it alone.

## Testing

- E2E is the default and sole testing mechanism unless the task explicitly requires otherwise.
- NEVER write unit tests after implementation. If isolated testing is truly necessary, identify the failure modes first and write the test before the implementation.
- Do not add tautological, implementation-mirroring, change-detector, or low-signal regression tests.
- Add a regression test for a bug fix only when it closes a genuine behavior-coverage gap.
- E2E verification should use a representative medium/hard scenario, not merely the easiest passing path.
- During development, run only targeted E2E checks when necessary. Run the full E2E suite near completion.

## Continuity Ledger (compaction-safe)

Maintain a single continuity file for this workspace: `CONTINUITY.md`.
`CONTINUITY.md` is the canonical briefing designed to survive compaction; do not rely on earlier chat/tool output unless it's reflected there.

### Operating rule

- Read `CONTINUITY.md` at the start of a new session/task, after context compaction/reset, or whenever prior state or decisions are uncertain.
- Update `CONTINUITY.md` only when there is a meaningful delta in: Goal/success criteria, Invariants/constraints, Decisions, State (Done/Now/Next), Open questions, Working set, or important tool outcomes.

### Keep it bounded (anti-bloat)

- Keep `CONTINUITY.md` short and high-signal:
  - `Snapshot`: ≤ 25 lines.
  - `Done (recent)`: ≤ 7 bullets.
  - `Working set`: ≤ 12 paths.
  - `Receipts`: keep last 10–20 entries.
- If sections exceed caps, compress older items into milestone bullets with pointers (commit/PR/log path/doc path). Do not paste raw logs.

### Anti-drift rules

- Facts only, no transcripts.
- Every entry must include:
  - a date or ISO timestamp (e.g., `2026-01-13` or `2026-01-13T09:42Z`)
  - a provenance tag: `[USER]`, `[CODE]`, `[TOOL]`, `[ASSUMPTION]`
- If unknown, write `UNCONFIRMED` (never guess). If something changes, supersede it explicitly (don't silently rewrite history).

### Decisions and incidents

- Record durable choices in `Decisions` as ADR-lite entries (e.g., `D001 ACTIVE: …`).
- For recurring weirdness, create a small, stable incident capsule (Symptoms / Evidence pointers / Mitigation / Status).

### Plan tool vs ledger

- Use `update_plan` for short-term execution scaffolding (3–7 steps).
- Use `CONTINUITY.md` for long-running continuity ("what/why/current state"), not micro task lists.
- Keep them consistent at the intent/progress level.
