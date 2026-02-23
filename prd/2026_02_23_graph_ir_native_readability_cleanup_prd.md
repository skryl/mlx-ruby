# GraphIR Native Monolithic Readability Cleanup PRD

## Status

Completed (2026-02-23)

## Context

`ext/mlx/graph_ir_native.cpp` remains intentionally monolithic and recently gained
`LoweringContext` + cast prelude helpers. The next maintainability bottleneck is still the
large ONNX lowering routine where repeated patterns remain:

1. repeated "known static shape required" validation,
2. repeated index-tensor cast-to-int64 setup,
3. repeated shape/dtype propagation from a single input,
4. long independent op branches that are logically mutually exclusive.

These can be refactored in-file without changing public API or behavior.

## Goals

1. Improve readability and maintainability of ONNX lowering while keeping monolithic layout.
2. Reduce repeated validation/casting boilerplate.
3. Make op-branch control flow easier to audit (single mutually-exclusive chain semantics).
4. Preserve exact external behavior and error classification semantics.

## Non-Goals

1. Splitting `graph_ir_native.cpp` into multiple files.
2. Expanding ONNX op coverage.
3. Changing Ruby public interfaces or exported native method signatures.
4. Any behavior-changing optimizations.

## Public API / Interface Changes

None. Internal refactor only.

## Phased Plan (Red/Green)

### Phase 0: Baseline Signal

#### Red

1. Run lowering-focused baseline tests:
   - `bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`

#### Green

1. Record baseline result for comparison.

#### Exit Criteria

1. Baseline lowering signal recorded.

### Phase 1: Shared Lowering Helpers

#### Red

1. Use baseline from Phase 0 as regression signal.

#### Green

1. Add helper to require known input shape with consistent unsupported message formatting.
2. Add helper to cast indices inputs to int64 (used by `Gather` and `GatherAxis`).
3. Add helper to assign inferred shape/dtype from one input tensor.
4. Replace duplicated code paths with these helpers.
5. Run targeted lowering tests:
   - `bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`

#### Exit Criteria

1. Duplicated validation/cast boilerplate is materially reduced.
2. Targeted lowering tests are green.

### Phase 2: Lowering Branch Readability Sweep

#### Red

1. Re-run targeted lowering tests if needed for immediate signal.

#### Green

1. Convert independent op checks into a mutually-exclusive `if / else if` chain where appropriate.
2. Extract unary passthrough-op detection helper for the large unary-op list.
3. Keep output metadata propagation unchanged.
4. Run targeted graph_ir regression checks:
   - `bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`
   - `bundle exec ruby -Itest test/graph_ir/graph_ir_webgpu_compat_report_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/export_onnx_compatibility_report_test.rb`

#### Exit Criteria

1. Control flow in `lower_onnx_node_default` is easier to scan and reason about.
2. Targeted checks are green.

### Phase 3: Full Regression Gate + Closeout

#### Red

1. Use prior phase outputs as comparison point.

#### Green

1. Run full suite:
   - `bundle exec rake test:all`
2. Mark PRD completed with test outcomes.

#### Exit Criteria

1. Full suite passes with no new regressions.
2. PRD checklist and status reflect completion.

## Test Gates

1. `bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`
2. `bundle exec ruby -Itest test/graph_ir/graph_ir_webgpu_compat_report_test.rb`
3. `bundle exec ruby -Itest test/graph_ir/export_onnx_compatibility_report_test.rb`
4. `bundle exec rake test:all`

## Acceptance Criteria

1. Monolithic file constraint is preserved.
2. Helper extraction removes repeated validation/casting patterns.
3. Lowering branch structure is clearer and still behavior-compatible.
4. No public API changes.
5. Targeted and full regression gates are green.

## Risks and Mitigations

1. Risk: subtle lowering behavior drift.
   - Mitigation: phase-scoped edits and mandatory lowering tests each phase.
2. Risk: error text drift can break compatibility tests.
   - Mitigation: keep unsupported-prefix semantics and reuse message formats.
3. Risk: broad regressions outside lowering.
   - Mitigation: final `rake test:all` gate.

## Implementation Checklist

- [x] Phase 0 Red: Run lowering baseline.
- [x] Phase 0 Green: Record baseline in execution log.
- [x] Phase 1 Green: Add shared lowering helpers and apply.
- [x] Phase 1 Green: Run lowering test gate.
- [x] Phase 2 Green: Branch readability sweep + unary helper.
- [x] Phase 2 Green: Run targeted graph_ir checks.
- [x] Phase 3 Green: Run `bundle exec rake test:all`.
- [x] Mark PRD `Completed (DATE)` with execution log.

## Execution Log

1. Phase 0 baseline (`bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`)
   - CPU: `62 runs, 426 assertions, 0 failures, 0 errors, 0 skips`
   - GPU: `62 runs, 426 assertions, 0 failures, 0 errors, 0 skips`
2. Phase 1/2 lowering gate (`bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`)
   - CPU: `62 runs, 426 assertions, 0 failures, 0 errors, 0 skips`
   - GPU: `62 runs, 426 assertions, 0 failures, 0 errors, 0 skips`
3. Phase 2 targeted compatibility checks
   - `bundle exec ruby -Itest test/graph_ir/graph_ir_webgpu_compat_report_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/export_onnx_compatibility_report_test.rb`
   - Result: both green (`0 failures, 0 errors`).
4. Phase 3 full-suite gate (`bundle exec rake test:all`)
   - CPU: `891 runs, 349956 assertions, 0 failures, 0 errors, 26 skips` (`Finished in 481.612472s`)
   - GPU: `891 runs, 349956 assertions, 0 failures, 0 errors, 26 skips` (`Finished in 478.033068s`)
