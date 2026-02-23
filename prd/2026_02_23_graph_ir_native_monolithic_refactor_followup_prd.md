# GraphIR Native Monolithic Refactor Follow-up PRD

## Status

Completed (2026-02-23)

## Context

`ext/mlx/graph_ir_native.cpp` is intentionally monolithic and already has sectioning, but
`lower_onnx_node_default` still carries high branching complexity and repeated bookkeeping
patterns (input casting prelude, output shape/dtype propagation, and node decode boilerplate).

A follow-up refactor should keep behavior unchanged while reducing local complexity and making
future ONNX lowering additions safer.

## Goals

1. Keep `ext/mlx/graph_ir_native.cpp` monolithic while improving internal structure.
2. Reduce repeated shape/dtype propagation code in the lowering path.
3. Introduce a lowering context object to reduce large parameter lists.
4. Deduplicate promoted-cast prelude handling across elementwise/comparison/select paths.
5. Preserve all public Ruby APIs and runtime semantics.

## Non-Goals

1. Splitting the file into multiple translation units.
2. Expanding ONNX operator coverage.
3. Changing Ruby-callable method signatures.
4. Performance tuning that alters behavior.

## Public API / Interface Changes

None. All changes are internal to `ext/mlx/graph_ir_native.cpp`.

## Phased Plan (Red/Green)

### Phase 0: Baseline Safety Signal

#### Red

1. Run targeted lowering regression tests to capture pre-change behavior.
   - `bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`

#### Green

1. Record baseline test result in execution log for comparison.

#### Exit Criteria

1. Targeted lowering baseline completed and recorded.

### Phase 1: Lowering Context + Shared Metadata Helpers

#### Red

1. Use Phase 0 baseline as the failure signal.

#### Green

1. Introduce an internal `LoweringContext` struct (initializers, used tensor names,
   known shapes, known dtypes).
2. Add helper functions for output shape/dtype propagation.
3. Add a parsed-node helper to decode op/op_type/inputs/outputs/arguments/attributes once.
4. Refactor call sites to use the context object while preserving behavior.
5. Run targeted lowering tests:
   - `bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`

#### Exit Criteria

1. Lowering function signatures are simplified with context usage.
2. Repeated output metadata loops are reduced.
3. Targeted lowering tests remain green.

### Phase 2: Deduplicate Promoted-Cast Prelude Flow

#### Red

1. Re-run targeted lowering tests before change for comparison if needed.

#### Green

1. Add a helper for the common promoted-cast + final-node append workflow.
2. Apply helper in:
   - arithmetic elementwise path (`Add`/`Subtract`/`Multiply`/`Divide`/`Maximum`/`Minimum`/`Power`)
   - comparison path (`Greater`/`Less`/`Equal`)
   - `Select`
3. Preserve special-case behavior (`Equal equal_nan`, fallback dtype rules).
4. Run targeted lowering tests:
   - `bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`

#### Exit Criteria

1. Promoted-cast flow duplication is materially reduced.
2. Targeted lowering tests remain green.

### Phase 3: Full Regression Gate + Closeout

#### Red

1. Use prior phase as baseline.

#### Green

1. Run full suite:
   - `bundle exec rake test:all`
2. Mark PRD complete with final results.

#### Exit Criteria

1. Full suite passes with no new regressions.
2. PRD status and checklist reflect completion.

## Test Gates

1. Targeted lowering tests for each refactor phase:
   - `bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`
2. Full suite completion gate:
   - `bundle exec rake test:all`

## Acceptance Criteria

1. `ext/mlx/graph_ir_native.cpp` remains monolithic.
2. `LoweringContext` and metadata helpers are adopted in lowering flow.
3. Promoted-cast prelude duplication is reduced via shared helper(s).
4. No public API changes.
5. Targeted lowering tests and full suite are green.

## Risks and Mitigations

1. Risk: semantic drift in lowering behavior.
   - Mitigation: phase-scoped changes and repeated lowering test gate.
2. Risk: accidental dtype/shape metadata propagation differences.
   - Mitigation: helper functions preserve existing behavior and are exercised by lowering tests.
3. Risk: broad regressions outside lowering.
   - Mitigation: mandatory `rake test:all` final gate.

## Implementation Checklist

- [x] Phase 0 Red: Run targeted lowering baseline.
- [x] Phase 0 Green: Record baseline in execution log.
- [x] Phase 1 Green: Introduce lowering context + metadata helpers.
- [x] Phase 1 Green: Run targeted lowering tests.
- [x] Phase 2 Green: Introduce promoted-cast shared helper and apply.
- [x] Phase 2 Green: Run targeted lowering tests.
- [x] Phase 3 Green: Run full suite (`bundle exec rake test:all`).
- [x] Mark PRD Completed with date and execution log.

## Execution Log

1. Phase 0 baseline (`bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`)
   - CPU: `62 runs, 426 assertions, 0 failures, 0 errors, 0 skips`
   - GPU: `62 runs, 426 assertions, 0 failures, 0 errors, 0 skips`
2. Phase 1/2 lowering gate (`bundle exec rake test TEST='test/graph_ir/*lowering_test.rb'`)
   - CPU: `62 runs, 426 assertions, 0 failures, 0 errors, 0 skips`
   - GPU: `62 runs, 426 assertions, 0 failures, 0 errors, 0 skips`
3. Additional targeted compatibility/export regression checks
   - `bundle exec ruby -Itest test/graph_ir/graph_ir_webgpu_compat_report_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/export_onnx_compatibility_report_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/export_onnx_binary_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/export_onnx_direct_test.rb`
   - Result: all green (`0 failures, 0 errors`).
4. Phase 3 full-suite gate (`bundle exec rake test:all`)
   - CPU: `891 runs, 349956 assertions, 0 failures, 0 errors, 26 skips` (`Finished in 469.718559s`)
   - GPU: `891 runs, 349956 assertions, 0 failures, 0 errors, 26 skips` (`Finished in 476.371518s`)
