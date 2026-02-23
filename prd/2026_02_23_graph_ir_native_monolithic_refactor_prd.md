# GraphIR Native Monolithic Refactor PRD

## Status

Completed (2026-02-23)

## Context

`ext/mlx/graph_ir_native.cpp` has grown to a single monolithic translation unit
with broad responsibilities:

1. Ruby/native interop.
2. GraphIR payload conversion and validation.
3. ONNX lowering and shape/dtype inference.
4. ONNX protobuf serialization.
5. Binary/external-data output writing.

The file is intentionally staying monolithic, but its internal structure and
large function boundaries need refactoring for readability, maintainability, and
safer incremental feature work.

## Goals

1. Improve internal organization and readability while keeping a single source file.
2. Reduce complexity in the largest hotspots without changing behavior.
3. Standardize internal utilities for recurring encode/validation patterns.
4. Keep runtime behavior and API compatibility unchanged.
5. Enforce stability by running full suite (`bundle exec rake test:all`) after
   every refactor phase.

## Non-Goals

1. Splitting `graph_ir_native.cpp` into multiple files.
2. Changing public Ruby APIs or native method signatures exposed to Ruby.
3. Expanding ONNX operator coverage.
4. Any performance tuning that changes semantic behavior.

## Public API / Interface Changes

None expected. This PRD targets internal refactoring only.

## Constraints

1. File must remain monolithic.
2. Refactors must be behavior-preserving.
3. Full test suite (including slow tests) must be run after each phase.
4. If baseline has existing unrelated failures, each phase must not introduce new failures.

## Phased Plan (Red/Green)

### Phase 0: Baseline + Safety Rails

#### Red

1. Run full suite and capture current baseline signal:
   - Command: `bundle exec rake test:all`
2. Record known failing tests (if any) as baseline-only debt.

#### Green

1. Land this PRD.
2. Confirm baseline signature is documented for comparison in later phases.

#### Exit Criteria

1. Baseline full-suite results recorded.
2. PRD committed in-repo.

### Phase 1: In-File Structural Sectioning

#### Red

1. Use baseline from Phase 0 as failure signal.

#### Green

1. Add explicit section banners in `ext/mlx/graph_ir_native.cpp` grouping:
   - Constants/types
   - Timing/logging
   - Ruby/JSON conversion
   - GraphIR lowering
   - Protobuf writers
   - Binary artifact writing
   - Ruby entrypoints
2. Keep behavior unchanged (comment-only + whitespace organization only).
3. Run full suite:
   - `bundle exec rake test:all`

#### Exit Criteria

1. Full suite run completed.
2. No new failures vs baseline.

### Phase 2: Targeted Hotspot Extraction (Lowering)

#### Red

1. Capture pre-change full-suite signal from previous phase.

#### Green

1. Refactor `lower_onnx_node_default` by extracting small helper functions for
   coherent subpaths (for example: cast/broadcast handling, gather/slice/scatter
   pre-processing, shape bookkeeping), while preserving exact semantics.
2. Keep all extracted helpers in the same file.
3. Run full suite:
   - `bundle exec rake test:all`

#### Exit Criteria

1. Largest lowering function reduced in size and complexity.
2. Full suite completed with no new failures vs baseline.

### Phase 3: Targeted Hotspot Extraction (Initializer + PB Tensor Encoding)

#### Red

1. Capture pre-change full-suite signal from previous phase.

#### Green

1. Refactor `tensor_raw_bytes_from_initializer` with local per-dtype helper
   functions to reduce branch complexity.
2. Refactor `pb_encode_tensor` to use dedicated helpers for inline/raw/external
   encoding branches (including complex64 float_data path).
3. Keep all helpers in the same file.
4. Run full suite:
   - `bundle exec rake test:all`

#### Exit Criteria

1. Initializer and tensor encode paths are easier to audit and test.
2. Full suite completed with no new failures vs baseline.

### Phase 4: Internal Consistency Sweep

#### Red

1. Capture pre-change full-suite signal from previous phase.

#### Green

1. Normalize recurring internal validation/error formatting patterns where low-risk.
2. Add concise internal comments only where control-flow is non-obvious.
3. Run full suite:
   - `bundle exec rake test:all`

#### Exit Criteria

1. Internal patterns are more consistent.
2. Full suite completed with no new failures vs baseline.
3. PRD marked completed.

## Test Gates

For each phase:

1. `bundle exec rake test:all` (mandatory).
2. If failures occur, compare with baseline and classify:
   - pre-existing baseline failure
   - new regression (must fix before phase completion)

## Acceptance Criteria

1. `ext/mlx/graph_ir_native.cpp` remains monolithic.
2. Planned hotspot refactors are completed.
3. No public API changes.
4. Full suite executed after each phase with no new regressions.
5. PRD checklist accurately reflects completion state.

## Risks and Mitigations

1. Risk: subtle semantic drift in lowering behavior.
   - Mitigation: phase-scoped changes plus mandatory full-suite gate each phase.
2. Risk: refactor merges become hard to review.
   - Mitigation: keep each phase narrow and behavior-preserving.
3. Risk: baseline unrelated failures obscure regressions.
   - Mitigation: explicit baseline tracking and per-phase diff of failures.

## Implementation Checklist

- [x] Phase 0 Red: Run and record baseline full-suite signal (`rake test:all`).
- [x] Phase 0 Green: Add PRD file and phase checklist.
- [x] Phase 1 Red: Confirm baseline signal for comparison.
- [x] Phase 1 Green: Add in-file section banners and run full suite.
- [x] Phase 2 Red: Capture pre-change full-suite signal.
- [x] Phase 2 Green: Refactor lowering hotspot and run full suite.
- [x] Phase 3 Red: Capture pre-change full-suite signal.
- [x] Phase 3 Green: Refactor initializer/PB tensor hotspot and run full suite.
- [x] Phase 4 Red: Capture pre-change full-suite signal.
- [x] Phase 4 Green: Consistency sweep and run full suite.
- [x] Mark PRD `Completed (DATE)` only when all phases are complete.

## Execution Log

1. Phase 0 Red baseline (`bundle exec rake test:all`)
   - Result: `891 runs, 349956 assertions, 0 failures, 0 errors, 26 skips`
   - Wall clock summary line: `Finished in 489.562728s`
2. Phase 1 Green gate (`bundle exec rake test:all`) after section-banner refactor
   - Result: `891 runs, 349956 assertions, 0 failures, 0 errors, 26 skips`
   - Wall clock summary line: `Finished in 472.479235s`
3. Phase 2 Green gate (`bundle exec rake test:all`) after lowering helper extraction
   - Result: `891 runs, 349956 assertions, 0 failures, 0 errors, 26 skips`
   - Wall clock summary line: `Finished in 523.696055s`
4. Phase 3 Green first gate attempt (`bundle exec rake test:all`) after initializer/tensor helper extraction
   - Result: `891 runs, 349925 assertions, 0 failures, 3 errors, 26 skips`
   - Errors: `GraphIrNativeTimingTest` native rebuild failed with missing `nlohmann/json.hpp` include path.
5. Phase 3 Green corrected gate (`bundle exec rake test:all`) after rerunning `ext/mlx/extconf.rb` and rebuilding native deps
   - Result: `891 runs, 349956 assertions, 0 failures, 0 errors, 26 skips`
   - Wall clock summary line: `Finished in 470.441464s`
6. Phase 4 Green gate (`bundle exec rake test:all`) after consistency sweep helpers
   - Result: `891 runs, 349956 assertions, 0 failures, 0 errors, 26 skips`
   - Wall clock summary line: `Finished in 496.723416s`
