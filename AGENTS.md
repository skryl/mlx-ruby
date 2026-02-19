# AGENTS.md

## Scope

Apply this workflow to all non-trivial tasks in this repo unless the user explicitly asks for a different process.

## Default Process

1. Understand the request and constraints.
2. Find existing design docs (`rfp/`, issue notes, prior PRDs) before creating new ones.
3. Use phased planning plus red/green testing for implementation.
4. Ship incrementally and keep status/checklists current.

## PRD/RFP Process

### When To Use A PRD/RFP

Create or update a PRD/RFP when work involves one or more of:

1. Multiple phases or milestones.
2. Cross-cutting changes across subsystems.
3. Non-obvious tradeoffs, risk, or unclear scope.
4. New test strategy, migration, or rollout criteria.

For very small, single-file fixes, a brief in-message plan is enough.

### Where To Put It

1. Store in `rfp/`.
2. Use naming: `YYYY_MM_DD_<topic>_rfp.md` (or `_prd.md` when explicitly requested).

### Required Sections

1. Status (`Proposed`, `In Progress`, `Completed` + date when completed).
2. Context.
3. Goals and non-goals.
4. Phased plan with clear red/green steps.
5. Exit criteria per phase.
6. Acceptance criteria for full completion.
7. Risks and mitigations.
8. Implementation checklist with phase checkboxes.

## Red/Green Execution Rules

For each phase:

1. **Red**
   - Add failing tests or a failing reproducible check first.
   - Capture the baseline failure signal.
2. **Green**
   - Implement the minimum change set needed to pass the new tests.
   - Re-run targeted tests and immediate regression checks.
3. **Refactor**
   - Refactor only after green.
   - Keep behavior unchanged during refactor.

Do not mark a phase complete until its exit criteria are met.

## Testing Gates

Run gates from specific to broad:

1. Unit/op-level tests for changed behavior.
2. Integration tests across touched boundaries.
3. End-to-end/runtime parity checks where relevant.
4. CI-facing or task-level command checks (rake tasks, linters, smoke scripts) for touched workflows.

If a gate cannot run locally, document exactly what was not run and why.

## Completion Criteria

Treat work as complete only when all are true:

1. Phase checklist items are updated.
2. New/updated tests are green.
3. Regressions in touched areas are checked.
4. PRD/RFP status reflects reality (`Completed` only when fully done).
