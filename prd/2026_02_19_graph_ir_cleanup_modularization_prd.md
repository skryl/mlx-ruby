# Graph IR Cleanup and Modularization PRD

## Status

Completed (February 19, 2026)

## Context

The current branch adds substantial Graph IR and ONNX/WebGPU functionality across:

1. `ext/mlx/native.cpp`
2. `lib/mlx/core.rb`
3. `lib/mlx/graph_ir.rb`

A review of `main...HEAD` identified two correctness risks and several maintainability issues:

1. `grad`/`value_and_grad` now cache mutable call state keyed by selection structure, which can be unsafe for concurrent/re-entrant calls.
2. `Convolution` lowering with `flip=true` in Graph IR can return before dtype metadata propagation is completed.
3. Graph IR responsibilities are spread across native C++ serialization, core wrappers, and a single large Ruby module (~2900 LOC), increasing drift risk and reducing testability.

## Goals

1. Fix correctness issues in gradient caching and `flip=true` convolution lowering.
2. Reduce coupling by centralizing Graph IR normalization/compatibility behavior under `MLX::GraphIR`.
3. Split Graph IR lowering/validation/inference concerns into smaller modules with stable public APIs.
4. Preserve behavior and public API compatibility for existing callers.
5. Add regression tests that lock in these fixes.

## Non-Goals

1. Expanding ONNX op coverage beyond current branch scope.
2. Redesigning the Graph IR schema (`ir_version` stays unchanged).
3. Rewriting the full native exporter pipeline in one step.
4. Large performance tuning unrelated to identified regressions.

## Phased Plan (Red/Green)

### Phase 0: Baseline and Repro Harness

#### Red

1. Add failing regression for concurrent/re-entrant `grad`/`value_and_grad` calls that share a selection key.
2. Add failing regression for `Convolution flip=true` export path ensuring output dtypes are available to downstream lowering.
3. Add characterization tests for current `export_graph_ir` normalization behavior and Core wrapper entry points.

#### Green

1. Ensure tests fail on baseline and capture precise failure signals.
2. Keep fixtures minimal and deterministic for repeatability.

#### Exit Criteria

1. Both correctness issues have reproducible failing tests.
2. Existing Graph IR wrapper behavior has characterization coverage before refactor.

### Phase 1: Correctness Fixes

#### Red

1. Keep Phase 0 failures active.
2. Add any additional unit checks discovered during implementation (e.g., nested/re-entrant transform invocation).

#### Green

1. Replace mutable shared `call_state` strategy in `build_grad_like_function` with invocation-safe state handling.
2. Update `flip=true` convolution lowering to propagate intermediate/output dtype metadata consistently.
3. Re-run targeted gradient and Graph IR/ONNX lowering tests.

#### Exit Criteria

1. New regression tests pass.
2. No failures in touched gradient and Graph IR conversion test suites.

### Phase 2: Graph IR Responsibility Consolidation

#### Red

1. Add failing or characterization tests around `MLX::Core` Graph IR normalization/backfill entry points.
2. Add contract tests for Graph IR normalization APIs moved under `MLX::GraphIR`.

#### Green

1. Move Graph-IR-specific normalization/backfill/dtype inference helpers out of `MLX::Core` and into `MLX::GraphIR` (or submodules under it).
2. Keep `MLX::Core` methods as thin delegating wrappers where public API must remain stable.
3. Preserve existing behavior and error messages unless explicitly improved and documented.

#### Exit Criteria

1. `MLX::Core` no longer owns schema-specific Graph IR internals.
2. Public API behavior remains compatible and test-covered.

### Phase 3: Graph IR File Modularization

#### Red

1. Add smoke tests asserting stable `validate!`, `to_onnx_stub`, and `webgpu_compatibility_report` behavior before code movement.
2. Add load tests for each extracted module file.

#### Green

1. Split `lib/mlx/graph_ir.rb` into focused files:
   - payload IO/normalization
   - validation/topology checks
   - ONNX lowering dispatch
   - dtype/shape inference and utility helpers
2. Keep `MLX::GraphIR` as the public facade with unchanged top-level method signatures.
3. Add lightweight internal docs for module boundaries and ownership.

#### Exit Criteria

1. `lib/mlx/graph_ir.rb` facade is significantly smaller and delegates to extracted modules.
2. All module load and behavior tests pass with unchanged public API.

### Phase 4: Native Serialization Boundary Hardening

#### Red

1. Add contract tests that compare expected Graph IR payload structure across export paths.
2. Add guard tests for schema drift points (tensor info keys, node argument encoding, constants encoding).

#### Green

1. Define and implement a single source-of-truth boundary for Graph IR payload assembly (minimizing duplicated schema logic between native C++ and Ruby).
2. Keep native side focused on trace extraction and primitive data capture.
3. Ensure serialization contract remains deterministic.

#### Exit Criteria

1. Drift-sensitive schema contract tests pass.
2. Duplicated schema logic is reduced and ownership is clear in code comments/docs.

## Exit Criteria by Phase Summary

1. Phase 0: Reproducible failing tests for both correctness bugs + characterization baselines.
2. Phase 1: Correctness fixes merged with regressions green.
3. Phase 2: Graph IR schema internals consolidated under `MLX::GraphIR`.
4. Phase 3: `graph_ir.rb` modularized with stable public API.
5. Phase 4: Native/Ruby serialization ownership clarified and drift-guarded by tests.

## Acceptance Criteria (Full Completion)

1. All phase checklists are updated and reflect actual status.
2. New regression tests for gradient state safety and `flip=true` dtype propagation are green.
3. Graph IR refactor tests (validation/lowering/compatibility/reporting) are green.
4. Touched integration checks pass (including ONNX export path tests and targeted smoke checks).
5. PRD status is updated to `Completed` with date only when all phases are done.

## Risks and Mitigations

1. Risk: Refactor changes break public API behavior.
   Mitigation: Characterization tests before movement; facade-level contract tests after movement.
2. Risk: Hidden dependencies in large lowering dispatcher cause regressions.
   Mitigation: Phase-by-phase extraction with op-family tests and narrow diffs.
3. Risk: Native/Ruby schema boundary changes introduce drift.
   Mitigation: Explicit serialization contract tests and documented ownership.
4. Risk: Concurrency bug fixes alter performance characteristics.
   Mitigation: Keep fixes minimal first; add follow-up benchmarking only if needed.

## Implementation Checklist

- [x] Phase 0: Add failing regressions for grad call-state safety and `flip=true` dtype propagation.
- [x] Phase 0: Add characterization coverage for current Graph IR normalization wrappers.
- [x] Phase 1: Implement invocation-safe gradient state handling.
- [x] Phase 1: Fix `flip=true` convolution dtype propagation in ONNX lowering.
- [x] Phase 1: Run targeted unit/integration tests for gradient + Graph IR paths.
- [x] Phase 2: Move Graph IR normalization/backfill internals from `MLX::Core` to `MLX::GraphIR`.
- [x] Phase 2: Keep `MLX::Core` delegating wrappers API-compatible.
- [x] Phase 3: Split `lib/mlx/graph_ir.rb` into submodules and keep a stable facade.
- [x] Phase 3: Add/refresh tests for extracted module boundaries.
- [x] Phase 4: Reduce schema duplication at the native/Ruby boundary.
- [x] Phase 4: Add/keep schema contract tests guarding against drift.
- [x] Final: Update PRD status to `Completed (DATE)` after all phases and tests are green.
