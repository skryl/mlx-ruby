# Core-Slim GraphIR/ONNX Refactor PRD

## Status

Completed (February 20, 2026)

## Context

Branch review for `main...HEAD` in `lib/mlx` and `ext/mlx` shows:

1. ~4.3k insertions across GraphIR/ONNX export paths, concentrated in:
   - `ext/mlx/native.cpp`
   - `lib/mlx/core.rb`
   - `lib/mlx/graph_ir/*.rb`
2. `MLX::Core` currently owns substantial GraphIR/ONNX orchestration and helper logic:
   - embedded Python ONNX builder script (`lib/mlx/core.rb`)
   - ONNX export + WebGPU harness assembly + smoke harness invocation
   - GraphIR normalization and ONNX option normalization helpers
3. Native extension currently performs GraphIR payload assembly + JSON generation + file writing in `core_export_graph_ir`, which creates schema duplication risk between C++ and Ruby.
4. ONNX lowering is mostly centralized in a single large dispatcher (`lib/mlx/graph_ir/onnx_stub.rb`), with many op-specific branches, which raises change risk and testing cost.
5. The same branch also includes non-IR `grad`/`value_and_grad` caching changes in `lib/mlx/core.rb`, which increases scope coupling.

## Goals

1. Remove `MLX::Core` GraphIR/ONNX facade methods so callers use `MLX::GraphIR` directly.
2. Move IR export/ONNX conversion implementation into modular `MLX::GraphIR` submodules.
3. Minimize schema/business logic in native C++ export boundary.
4. Decompose ONNX lowering into stable, testable units by op family/pass.
5. Preserve public API behavior and payload schema compatibility.

## Non-Goals

1. Expanding ONNX op coverage beyond current behavior.
2. Changing GraphIR schema version (`ir_version` remains unchanged).
3. Large runtime/performance tuning outside modularization goals.
4. Reworking unrelated autograd behavior as part of IR/ONNX refactor.

## Phased Plan (Red/Green)

### Phase 0: Contract Freeze and Baseline

#### Red

1. Add characterization tests for current public entrypoints:
   - `export_graph_ir_json`
   - `validate_graph_ir`
   - `graph_ir_to_onnx_stub`
   - `graph_ir_to_onnx_json`
   - `export_onnx_json`
   - `onnx_json_to_onnx`
   - `export_onnx_webgpu_harness`
2. Add schema contract tests covering GraphIR payload fields, tensor metadata, node argument encoding, and ONNX stub shape/dtype fields.
3. Add fixture-based behavior snapshots for external-data ONNX export and harness manifest generation.

#### Green

1. Confirm red tests fail when contracts are intentionally perturbed.
2. Establish stable baselines to protect the refactor.

#### Exit Criteria

1. Public GraphIR/ONNX APIs have characterization coverage.
2. Payload and ONNX stub contracts are frozen by tests.

### Phase 1: Core Facade Slimming

#### Red

1. Add tests asserting `MLX::Core` GraphIR/ONNX methods remain API-compatible while delegating behavior.
2. Add regression checks for existing error messages/argument validation where compatibility matters.

#### Green

1. Extract GraphIR/ONNX implementation from `lib/mlx/core.rb` into new modules:
   - `lib/mlx/graph_ir/exporter.rb`
   - `lib/mlx/graph_ir/onnx/exporter.rb`
   - `lib/mlx/graph_ir/webgpu_harness.rb`
2. Move helper methods from `MLX::Core` to these modules:
   - ONNX external data option normalization
   - JSON compatibility conversion delegation points
   - harness input/example and asset copy helpers
3. Remove `MLX::Core` GraphIR/ONNX wrappers/delegators and migrate callsites to `MLX::GraphIR`.

#### Exit Criteria

1. `MLX::Core` no longer exposes GraphIR/ONNX export facade methods.
2. Behavior remains test-equivalent to Phase 0 baselines.

### Phase 2: ONNX Builder Isolation

#### Red

1. Add tests for Python invocation contract (args, external data options, failure surface).
2. Add tests that fail if ONNX builder script location/entrypoint changes without adapter updates.

#### Green

1. Move embedded Python script constant out of `lib/mlx/core.rb` into a dedicated builder layer:
   - `lib/mlx/graph_ir/onnx/python_builder.rb` and/or a dedicated script file.
2. Introduce `MLX::GraphIR::ONNX::PythonBuilder` with a single public build method.
3. Ensure core/facade layer no longer embeds large script literals.

#### Exit Criteria

1. ONNX build execution is isolated behind one adapter boundary.
2. `lib/mlx/core.rb` no longer contains embedded ONNX Python source.

### Phase 3: ONNX Lowering Modularization

#### Red

1. Add op-family characterization tests for lowering outputs (node op_type, attrs, shape/dtype inferences, aux initializers).
2. Add compatibility report regression tests for WebGPU operator support output.

#### Green

1. Split `lib/mlx/graph_ir/onnx_stub.rb` into focused units:
   - `lib/mlx/graph_ir/onnx/lowerer.rb` (pipeline orchestration)
   - `lib/mlx/graph_ir/onnx/op_type_resolver.rb`
   - `lib/mlx/graph_ir/onnx/lowering/elementwise.rb`
   - `lib/mlx/graph_ir/onnx/lowering/reduction.rb`
   - `lib/mlx/graph_ir/onnx/lowering/shape_ops.rb`
   - `lib/mlx/graph_ir/onnx/lowering/convolution.rb`
   - `lib/mlx/graph_ir/onnx/lowering/indexing.rb`
2. Replace large case-driven lowering with table/registry dispatch by op family.
3. Keep `MLX::GraphIR.to_onnx_stub` and `MLX::GraphIR.webgpu_compatibility_report` signatures unchanged.

#### Exit Criteria

1. Lowering implementation is modularized by responsibility/op family.
2. Public stub and compatibility APIs remain stable.

### Phase 4: Native Export Boundary Slimming

#### Red

1. Add contract tests for native-exported records and final GraphIR payload equivalence.
2. Add deterministic output tests comparing repeated exports.

#### Green

1. Reduce native responsibility to trace capture + typed record emission.
2. Move final GraphIR schema assembly and JSON serialization ownership to Ruby GraphIR exporter modules.
3. Keep backward-compatible `MLX::GraphIR.export_graph_ir_json` payload/schema behavior.

#### Exit Criteria

1. Native side does not own GraphIR JSON schema details beyond raw capture fields.
2. Ruby GraphIR layer is single source of truth for payload assembly/normalization.

### Phase 5: Scope Hygiene for Non-IR Core Changes

#### Red

1. Add/retain regression tests for `grad`/`value_and_grad` caching behavior currently changed in this branch.

#### Green

1. Separate non-IR autograd changes from IR/ONNX refactor path (separate PR/commit stream).
2. Keep IR/ONNX refactor diffs focused to reduce review risk and rollback blast radius.

#### Exit Criteria

1. IR/ONNX modularization can be reviewed independently from autograd changes.

### Phase 6: Full Docs and README Update (Post-Refactor)

#### Red

1. Add/refresh doc coverage checks that fail when GraphIR/ONNX public APIs and docs drift.
2. Add failing assertions for stale README usage examples and outdated module ownership notes.

#### Green

1. Update top-level `README` and any GraphIR/ONNX-specific README sections to match final refactored API boundaries and workflows.
2. Update docs pages for:
   - GraphIR export flow
   - ONNX conversion flow
   - WebGPU harness generation/smoke workflow
   - module ownership (`MLX::Core` facade vs `MLX::GraphIR` internals)
3. Refresh code examples/commands so they are runnable and consistent with the refactor output paths/options.

#### Exit Criteria

1. README and docs reflect the refactored architecture and public entrypoints.
2. No known drift between examples/documentation and tested behavior.

## Exit Criteria by Phase Summary

1. Phase 0: Contracts and snapshots established.
2. Phase 1: Core becomes thin facade for IR/ONNX operations.
3. Phase 2: ONNX Python builder isolated from core.
4. Phase 3: ONNX lowering decomposed into modular units.
5. Phase 4: Native boundary narrowed and schema ownership clarified.
6. Phase 5: Non-IR core changes separated from IR/ONNX refactor scope.
7. Phase 6: README/docs fully updated and aligned after refactor.

## Acceptance Criteria (Full Completion)

1. `MLX::Core` no longer carries GraphIR/ONNX export facade methods.
2. GraphIR export + ONNX conversion logic lives under modular `lib/mlx/graph_ir/**`.
3. Native exporter no longer duplicates high-level schema assembly logic.
4. README and docs are updated for the finalized refactored workflows and boundaries.
5. Characterization + regression tests for touched behavior are green.
6. PRD checklist and status reflect actual implementation progress.

## Risks and Mitigations

1. Risk: API drift while moving logic out of `MLX::Core`.
   Mitigation: Characterization tests and compatibility snapshots before extraction.
2. Risk: Lowering regressions during dispatcher decomposition.
   Mitigation: Op-family contract tests with fixture comparisons.
3. Risk: Native/Ruby contract mismatch after boundary slimming.
   Mitigation: Deterministic schema contract tests at native-output and final-payload boundaries.
4. Risk: Scope creep from unrelated core changes.
   Mitigation: Explicit phase for scope hygiene and separate review streams.

## Implementation Checklist

- [x] Phase 0 red: add characterization tests for GraphIR/ONNX entrypoints.
- [x] Phase 0 green: establish payload/stub contract fixtures.
- [x] Phase 1 red: add facade-compatibility tests for `MLX::Core`.
- [x] Phase 1 green: extract GraphIR/ONNX logic into `MLX::GraphIR` exporter modules.
- [x] Phase 1 green: keep `MLX::Core` wrappers thin and API-compatible.
- [x] Phase 2 red: add ONNX builder invocation/error contract tests.
- [x] Phase 2 green: isolate Python builder adapter and remove embedded script from core.
- [x] Phase 3 red: add op-family lowering characterization tests.
- [x] Phase 3 green: split ONNX lowering into modular files with dispatch registry.
- [x] Phase 4 red: add native-output to payload-equivalence contract tests.
- [x] Phase 4 green: narrow native export responsibilities and centralize schema assembly in Ruby.
- [x] Phase 5 red: retain coverage for non-IR autograd changes.
- [x] Phase 5 green: separate non-IR changes from IR/ONNX modularization stream.
- [x] Phase 6 red: add docs/README drift checks and stale-example guards.
- [x] Phase 6 green: complete full docs and README refresh for refactored GraphIR/ONNX flows.
- [x] Final: update PRD status to `Completed (DATE)` only when all phases and tests are complete.
