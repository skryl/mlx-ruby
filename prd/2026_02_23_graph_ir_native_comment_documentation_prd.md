# GraphIR Native Comment Documentation PRD

## Status

Completed (2026-02-23)

## Context

`ext/mlx/graph_ir_native.cpp` is intentionally monolithic and already sectioned, but most
logic is self-contained without explanatory comments. The file is large (~5.7k LOC) and spans
multiple domains (Ruby interop, GraphIR capture, ONNX lowering, protobuf encoding, binary I/O),
which makes onboarding and maintenance slower.

This PRD documents a behavior-preserving pass to add thorough, high-value comments across the
file while keeping it monolithic.

## Goals

1. Add comprehensive explanatory comments for architecture, data flow, and non-obvious logic.
2. Document key invariants and assumptions in lowering and protobuf encoding.
3. Keep behavior and APIs unchanged.
4. Preserve readability (comment useful intent/tradeoffs; avoid noisy line-by-line narration).

## Non-Goals

1. Splitting the file into smaller translation units.
2. Refactoring logic or adding/removing ONNX op coverage.
3. Changing Ruby/native public APIs.
4. Performance tuning.

## Public API / Interface Changes

None.

## Phased Plan (Red/Green)

### Phase 0: Baseline Signal

#### Red

1. Run targeted lowering baseline:
   - `bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`

#### Green

1. Record baseline for comparison.

#### Exit Criteria

1. Baseline signal captured.

### Phase 1: File- and Section-Level Documentation

#### Red

1. Use Phase 0 baseline as guardrail.

#### Green

1. Add top-of-file architecture overview comment.
2. Add concise section-level comments clarifying responsibilities and handoff boundaries.
3. Add comments for core internal data structures used across sections.
4. Run targeted lowering tests:
   - `bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`

#### Exit Criteria

1. Major sections have clear purpose/invariant comments.
2. Targeted tests remain green.

### Phase 2: Complex Function Documentation

#### Red

1. Re-use prior test signal for regression checks.

#### Green

1. Add comments to non-obvious functions, including:
   - export callback aggregation and state capture conversion,
   - JSON source parsing and conversion boundaries,
   - lowering control flow and shape/dtype tracking,
   - protobuf encoding decisions (inline vs external data, complex64 handling),
   - compatibility report simulation strategy.
2. Keep comments accurate to existing behavior.
3. Run targeted graph_ir checks:
   - `bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`
   - `bundle exec ruby -Itest test/graph_ir/graph_ir_webgpu_compat_report_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/export_onnx_compatibility_report_test.rb`

#### Exit Criteria

1. Tricky/critical functions have explanatory comments.
2. Targeted checks are green.

### Phase 3: Final Regression Gate + Closeout

#### Red

1. Use prior phase as baseline.

#### Green

1. Run full suite:
   - `bundle exec rake test:all`
2. Update PRD status/checklist to completed with execution log.

#### Exit Criteria

1. Full suite passes with no new regressions.
2. PRD accurately reflects completion.

## Test Gates

1. `bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`
2. `bundle exec ruby -Itest test/graph_ir/graph_ir_webgpu_compat_report_test.rb`
3. `bundle exec ruby -Itest test/graph_ir/export_onnx_compatibility_report_test.rb`
4. `bundle exec rake test:all`

## Acceptance Criteria

1. `ext/mlx/graph_ir_native.cpp` remains monolithic.
2. Architecture and major control-flow paths are thoroughly documented with comments.
3. No behavior or public API changes.
4. Targeted regression gates are green; full-suite gate may be waived for comment-only work when explicitly directed by the user.

## Risks and Mitigations

1. Risk: comments become stale or misleading.
   - Mitigation: keep comments factual and local to current behavior/invariants.
2. Risk: over-commenting harms readability.
   - Mitigation: prioritize non-obvious logic and boundaries over trivial statements.
3. Risk: accidental code changes during comment pass.
   - Mitigation: comment-only edits plus mandatory test gates.

## Implementation Checklist

- [x] Phase 0 Red: Run targeted lowering baseline.
- [x] Phase 0 Green: Record baseline in execution log.
- [x] Phase 1 Green: Add file/section-level comments.
- [x] Phase 1 Green: Run lowering tests.
- [x] Phase 2 Green: Add complex-function comments.
- [x] Phase 2 Green: Run targeted graph_ir checks.
- [x] Phase 3 Green: Run full suite (`bundle exec rake test:all`) - waived by explicit user direction for comment-only phase.
- [x] Mark PRD Completed with execution log.

## Execution Log

1. Phase 0 baseline (`bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`)
   - CPU: `62 runs, 426 assertions, 0 failures, 0 errors, 0 skips`
   - GPU: `62 runs, 426 assertions, 0 failures, 0 errors, 0 skips`
2. Post-comment Phase 1 regression (`bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`)
   - CPU: `62 runs, 426 assertions, 0 failures, 0 errors, 0 skips`
   - GPU: `62 runs, 426 assertions, 0 failures, 0 errors, 0 skips`
3. Post-comment Phase 2 targeted checks
   - `bundle exec ruby -Itest test/graph_ir/graph_ir_webgpu_compat_report_test.rb`
     - `2 runs, 8 assertions, 0 failures, 0 errors, 0 skips`
   - `bundle exec ruby -Itest test/graph_ir/export_onnx_compatibility_report_test.rb`
     - `2 runs, 12 assertions, 0 failures, 0 errors, 0 skips`
4. Phase 3 full suite gate
   - Waived by explicit user instruction: "you don't need to run tests for commenting phases".
