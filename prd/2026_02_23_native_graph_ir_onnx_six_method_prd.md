# Native GraphIR/ONNX Six-Method API PRD

## Status

Completed (2026-02-23)

## Context

Current `MLX::GraphIR` exposes a mixed public surface that includes validation,
compatibility reporting, JSON/binary conversion split across Ruby and native
layers, and WebGPU harness orchestration. The branch goal is a clean-break API
for upstreaming and straightforward Python binding support:

1. Keep a compact native-backed public GraphIR method surface.
2. Make GraphIR/ONNX runtime logic native-first.
3. Keep JSON methods only for debugging, with binary ONNX as primary output.
4. Remove legacy public methods and legacy runtime code paths.
5. Add parity tests that compare native ONNX binary output against the Python
   oracle builder.

## Goals

1. Replace `MLX::GraphIR` public facade with exactly:
   - `export_onnx`
   - `export_onnx_json`
   - `export_graph_ir`
   - `export_graph_ir_json`
   - `graph_ir_to_onnx`
   - `graph_ir_to_onnx_json`
2. Implement end-to-end native execution for all public GraphIR methods.
3. Remove `onnx_json_to_onnx`, `validate!`, `compatibility_report`,
   `export_onnx_webgpu_harness`, and `smoke_test_onnx_webgpu_harness` from the
   `MLX::GraphIR` public facade.
4. Add test-only Python oracle parity tests for native ONNX binary output.

## Non-Goals

1. Expanding ONNX operator coverage beyond current lowering behavior.
2. Migrating WebGPU harness internals to native.
3. Preserving legacy GraphIR method compatibility.

## Public API Changes

### Canonical Public Surface

1. `MLX::GraphIR.export_onnx(target_path, fun, *extras, shapeless:, opset:, model_name:, external_data:, external_data_file:, external_data_size_threshold:, **trace_kwargs)`
2. `MLX::GraphIR.export_onnx_json(fun, *extras, shapeless:, opset:, model_name:, **trace_kwargs)`
3. `MLX::GraphIR.export_graph_ir(fun, *extras, shapeless:, **trace_kwargs)`
4. `MLX::GraphIR.export_graph_ir_json(fun, *extras, shapeless:, **trace_kwargs)`
5. `MLX::GraphIR.graph_ir_to_onnx(target_path, graph_ir_source, opset:, model_name:, external_data:, external_data_file:, external_data_size_threshold:)`
6. `MLX::GraphIR.graph_ir_to_onnx_json(graph_ir_source, opset:, model_name:)`
7. `MLX::GraphIR.export_onnx_compatibility_report(fun, *extras, shapeless:, **trace_kwargs)`

### Removed From Public Facade

1. `MLX::GraphIR.validate!`
2. `MLX::GraphIR.compatibility_report`
3. `MLX::GraphIR.onnx_json_to_onnx`
4. `MLX::GraphIR.export_onnx_webgpu_harness`
5. `MLX::GraphIR.smoke_test_onnx_webgpu_harness`

## Phased Plan (Red/Green)

### Phase 0: Contract Lock + PRD

#### Red

1. Add failing facade contract tests for exact public method surface.
2. Add failing tests that removed methods are not public.

#### Green

1. Add this PRD and wire phase checklist.
2. Implement/adjust API contract tests to pass with new method surface.

#### Exit Criteria

1. Public surface is test-locked.

### Phase 1: Native GraphIR Export Surface

#### Red

1. Add failing tests for `export_graph_ir` Hash return and
   `export_graph_ir_json` parity against hash serialization.
2. Add failing tests for `shapeless:` behavior in new GraphIR export methods.

#### Green

1. Add native method returning GraphIR payload as Ruby Hash.
2. Route `MLX::GraphIR.export_graph_ir` and
   `MLX::GraphIR.export_graph_ir_json` through native only.

#### Exit Criteria

1. Both GraphIR export methods are native-backed and green.

### Phase 2: Native GraphIR Source Decoding + ONNX JSON

#### Red

1. Add failing tests for `graph_ir_to_onnx_json` with:
   - Hash source
   - JSON string source
   - file-path source
2. Add failing tests for unsupported-op translation behavior.

#### Green

1. Implement native source decoding (`Hash`/JSON string/path/IO-like).
2. Route `MLX::GraphIR.graph_ir_to_onnx_json` through native source decoding.

#### Exit Criteria

1. ONNX JSON conversion accepts all supported source forms natively.

### Phase 3: Native ONNX Binary Writer

#### Red

1. Add failing tests for:
   - `export_onnx` writes valid ONNX
   - `graph_ir_to_onnx` writes valid ONNX
   - external data options behavior
2. Add failing tests replacing `onnx_json_to_onnx` usage in touched helpers.

#### Green

1. Implement native ONNX binary serialization and path writing.
2. Route both binary methods entirely through native.
3. Remove Python builder from production path.

#### Exit Criteria

1. Binary ONNX export path no longer depends on Python runtime in production.

### Phase 4: Python Oracle Binary Parity

#### Red

1. Add failing test-only parity suite that builds ONNX from the same stub using:
   - Python oracle builder
   - native binary writer
2. Compare model summaries (opset, graph topology, initializer signatures) and
   ONNX checker validity.

#### Green

1. Stabilize native serializer to pass parity suite.
2. Keep Python builder usage restricted to tests only.

#### Exit Criteria

1. Native-vs-Python binary parity tests are green.

### Phase 5: Legacy Facade Removal + Callsite Migration

#### Red

1. Add failing drift/contract tests to forbid removed facade methods.
2. Add failing tests for updated internal task/script callsites.

#### Green

1. Rewrite `lib/mlx/graph_ir.rb` to native-backed public methods only.
2. Migrate in-repo callsites/tests/docs away from removed methods.
3. Remove dead runtime code paths that support removed public methods.

#### Exit Criteria

1. No removed methods remain on `MLX::GraphIR.public_methods(false)`.
2. Touched callsites are migrated and tested.

## Test Gates

1. Unit/contract tests for `MLX::GraphIR` method surface.
2. GraphIR export/ONNX conversion parity tests for touched phases.
3. Native ONNX binary validity checks (python `onnx` checker where available).
4. Native-vs-Python oracle parity tests for binary output.
5. Drift tests for docs/public method contract updates.

If a gate cannot run locally, document exactly which command/test was skipped
and why.

## Acceptance Criteria

1. `MLX::GraphIR` public methods match the contract in this PRD.
2. GraphIR export and GraphIR->ONNX JSON paths are native-backed.
3. ONNX binary export/conversion paths are native-backed.
4. Python builder is not used by runtime export code.
5. Native-vs-Python ONNX binary parity tests are green.
6. Phase checklist reflects completed status accurately.

## Risks and Mitigations

1. Risk: Native serializer mismatch with Python ONNX behavior.
   Mitigation: dedicated oracle parity tests before removing runtime Python path.
2. Risk: Broad API break causes internal script/test churn.
   Mitigation: phase-by-phase callsite migration with contract tests.
3. Risk: Incomplete source decoding support (Hash/JSON/path).
   Mitigation: source-form matrix tests in Phase 2.

## Implementation Checklist

- [x] Phase 0 Red: Add failing facade contract tests.
- [x] Phase 0 Green: Land PRD and pass updated facade contract tests.
- [x] Phase 1 Red: Add failing `export_graph_ir`/`export_graph_ir_json` parity tests.
- [x] Phase 1 Green: Implement native GraphIR Hash + JSON export methods.
- [x] Phase 2 Red: Add failing source-form tests for `graph_ir_to_onnx_json`.
- [x] Phase 2 Green: Implement native Hash/JSON/path decoding for ONNX JSON conversion.
- [x] Phase 3 Red: Add failing binary export tests for `export_onnx` and `graph_ir_to_onnx`.
- [x] Phase 3 Green: Implement native ONNX binary writer and runtime path migration.
- [x] Phase 4 Red: Add failing Python oracle parity tests for ONNX binary output.
- [x] Phase 4 Green: Make native binary writer pass parity suite.
- [x] Phase 5 Red: Add failing tests for removed legacy methods/callsites.
- [x] Phase 5 Green: Remove legacy public facade methods and migrate touched callsites.
- [x] Phase 5 Green: Delete remaining legacy Ruby runtime files under
  `lib/mlx/graph_ir` (`constants.rb`, `exporter.rb`, `payload.rb`,
  `validation.rb`, `onnx/*`), keeping only the public facade and active
  `WebGPUHarness`.
- [x] Post-completion: Add `export_onnx_compatibility_report` as native-backed
  public preflight API for traced models.
- [x] Run mandatory test gates for all touched areas.
- [x] Mark PRD `Completed (DATE)` only when all phases are complete.

## Validation Summary

Executed and green:

1. Facade and conversion parity: phases `281`, `284`, `285`, `286`, `289`, `295`, `305`, `307`, `329`, `331`, `332`, `333`, `334`, `336`, `337`.
2. WebGPU harness/runtime parity: phases `309`, `310`, `311`, `313`, `314`.
3. Task-level guardrails: `test/tasks/web_demo_weights_guardrail_test.rb`, `test/tasks/web_assets_task_test.rb`.

Expected environment skips encountered in integration coverage:

1. `phase313`: one skip due optional environment/runtime condition.
2. `phase314`: one skip due optional environment/runtime condition.
3. `phase324`: skipped due optional submodule/runtime prerequisites.
