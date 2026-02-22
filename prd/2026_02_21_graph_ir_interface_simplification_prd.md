# GraphIR Interface Simplification PRD

## Status

Completed (February 21, 2026)

## Context

Current `MLX::GraphIR` API surface has drifted into a mix of:

1. Core user-facing methods for GraphIR/ONNX/WebGPU workflows.
2. Redundant overlap (`graph_ir_to_onnx_json` + `graph_ir_to_onnx_payload`).
3. Accidental public helpers exposed via module-level utility methods.
4. Inconsistent behavior contracts (for example return type differences by target type).
5. Native boundary quirks (boolean sentinel parsing and uneven error translation).

This effort intentionally does **not** preserve legacy usage compatibility. The goal is to simplify and harden the current interface in place.

## Goals

1. Define a small, explicit, stable `MLX::GraphIR` API.
2. Remove redundant helpers and accidental public methods.
3. Make behavior contracts deterministic (arguments, return types, errors).
4. Align Ruby facade and native boundary behavior.
5. Keep docs/tests as contract guards for the final interface.

## Non-Goals

1. Expanding ONNX operator lowering coverage.
2. Changing GraphIR schema version.
3. Introducing a separate versioned namespace (for example `v2`).
4. Adding compatibility/deprecation shims for removed legacy paths.

## Public API and Interface Changes

### Keep (final supported facade)

1. `export_graph_ir_json`
2. `validate!`
3. `compatibility_report`
4. `graph_ir_to_onnx_json`
5. `export_onnx_json` (add explicit `shapeless:` keyword)
6. `onnx_json_to_onnx` (path-like target only, deterministic return contract)
7. `export_onnx_webgpu_harness`
8. `smoke_test_onnx_webgpu_harness`

### Remove from public facade

1. `graph_ir_to_onnx_payload`
2. `compatibility_report_json`
3. `load_payload`
4. `dump_json`
5. `onnx_json_compatible_value`

### Native boundary policy

1. Treat `MLX::GraphIR::Native` as internal implementation details.
2. Remove public-path dependence on trailing-boolean sentinel argument parsing.
3. Standardize typed exception mapping behavior across ONNX conversion/report endpoints.

## Phased Plan with Red/Green Steps

### Phase 1: Facade Contract Simplification

#### Red

1. Add failing API contract tests for:
   - final supported public method set,
   - removed method absence,
   - `export_onnx_json(..., shapeless: true)` support.

#### Green

1. Update `MLX::GraphIR` facade methods/signatures.
2. Route `shapeless:` through to ONNX exporter.
3. Remove non-target methods from public facade.

#### Refactor

1. Consolidate shared internal helper paths after contract is green.

#### Exit Criteria

1. Facade contract tests pass.
2. Removed methods are no longer publicly callable.
3. `shapeless:` passthrough is tested and passing.

### Phase 2: ONNX Writer Determinism

#### Red

1. Add failing tests for `onnx_json_to_onnx` path-only contract and deterministic return.

#### Green

1. Enforce path-like target only.
2. Remove IO-target behavior branching.
3. Return deterministic success value for all valid calls.

#### Refactor

1. Normalize path and external data helper logic.

#### Exit Criteria

1. Writer contract tests pass for normal and external-data paths.

### Phase 3: Native Boundary Hardening

#### Red

1. Add failing tests for:
   - unsupported error translation consistency,
   - elimination of public ambiguity from trailing boolean parsing.

#### Green

1. Update native invocation parsing/entrypoint usage for explicit contract shape.
2. Standardize native exception translation path for compatibility report and ONNX conversion methods.

#### Refactor

1. Collapse duplicated native exception plumbing.

#### Exit Criteria

1. Native-boundary contract tests pass.

### Phase 4: Docs and Drift Alignment

#### Red

1. Add failing docs drift tests for final method list and removed methods.

#### Green

1. Update README and docs to final API and flow.
2. Update parity drift checks to enforce the final list.

#### Refactor

1. Remove stale wording and duplicate API references.

#### Exit Criteria

1. Docs and drift tests pass and match implemented surface.

## Exit Criteria Per Phase

1. Phase 1: Public facade simplified and enforced by tests.
2. Phase 2: ONNX write contract deterministic.
3. Phase 3: Native API boundary behavior normalized.
4. Phase 4: Documentation/tests fully aligned with implementation.

## Acceptance Criteria

1. `MLX::GraphIR` public singleton methods match final supported facade exactly.
2. Removed methods are not publicly callable and not documented as supported.
3. `export_onnx_json` exposes and honors `shapeless:`.
4. `onnx_json_to_onnx` path-only contract is enforced with deterministic return behavior.
5. Native unsupported/error translation behavior is consistent for relevant endpoints.
6. Docs and drift tests enforce and describe the same API surface.
7. Changed behavior is covered by targeted unit/integration/parity checks.

## Risks and Mitigations

1. Risk: broad in-place break introduces downstream disruption.
   Mitigation: explicit release notes and strict API contract tests.
2. Risk: hidden internal callers depend on removed methods.
   Mitigation: repo-wide usage scans before each removal and targeted updates.
3. Risk: native error handling regressions.
   Mitigation: dedicated native boundary tests for each exported entrypoint.
4. Risk: docs drift after rapid interface changes.
   Mitigation: drift tests updated as part of each phase.

## Implementation Checklist

- [x] Phase 1 Red: Add failing facade API contract tests.
- [x] Phase 1 Green: Apply facade method/signature changes and remove redundant methods.
- [x] Phase 1 Refactor: Internal helper consolidation.
- [x] Phase 2 Red: Add failing deterministic writer contract tests.
- [x] Phase 2 Green: Enforce path-only writer behavior and deterministic return.
- [x] Phase 2 Refactor: Simplify writer helper paths.
- [x] Phase 3 Red: Add failing native-boundary error/contract tests.
- [x] Phase 3 Green: Normalize native parsing and exception translation behavior.
- [x] Phase 3 Refactor: Consolidate native exception plumbing.
- [x] Phase 4 Red: Add failing docs/drift contract tests for final API.
- [x] Phase 4 Green: Update README/docs/drift tests to final interface.
- [x] Phase 4 Refactor: Cleanup stale references and duplicate wording.
- [x] Run targeted test gates for all touched phases.
- [x] Mark PRD `Completed (DATE)` only when all criteria are met.

## Implementation Progress

1. Added PRD and moved status to `In Progress`.
2. Phase 1 red/green slice completed for `export_onnx_json` facade keyword support:
   - Added failing contract test `test/parity/phase334_export_onnx_shapeless_facade_parity_test.rb`.
   - Updated facade signature in `lib/mlx/graph_ir.rb` to accept and forward `shapeless:`.
   - Verified the new test passes after the change.
3. Phase 1 red/green slice completed for public `compatibility_report_json` removal:
   - Extended `test/parity/phase334_export_onnx_shapeless_facade_parity_test.rb` with a failing surface-contract assertion.
   - Removed `MLX::GraphIR.compatibility_report_json` from `lib/mlx/graph_ir.rb`.
   - Verified the parity contract test and a direct export parity regression test pass.
4. Phase 1 completed:
   - Removed `MLX::GraphIR.graph_ir_to_onnx_payload` from facade.
   - Made `load_payload`, `dump_json`, and `onnx_json_compatible_value` non-public.
   - Added `TestSupport.parse_onnx_stub` and migrated parity tests to `graph_ir_to_onnx_json`.
5. Phase 2 completed:
   - Enforced path-like-only targets in `onnx_json_to_onnx`.
   - Removed IO target branch and standardized return value to written path string.
   - Updated ONNX parity tests for deterministic writer contract and IO rejection.
6. Phase 3 completed:
   - Refactored native GraphIR payload capture into a dedicated helper.
   - Updated native ONNX export path to avoid trailing-boolean sentinel dependency.
   - Unified native compatibility-report exception translation through shared handler.
7. Phase 4 completed:
   - Updated README/docs to remove `graph_ir_to_onnx_payload` references and reflect path-only ONNX writer.
   - Updated docs drift parity expectations for final public flow terms.
