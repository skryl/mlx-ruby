# PRD: MLX Computation Graph IR Export and ONNX/WebGPU Conversion

## 1. Summary

Implement a production-grade path to export MLX computation graphs into an external IR, then convert that IR into ONNX artifacts that can run in browser environments via WebGPU runtimes.

This PRD defines:

- A canonical external graph IR for MLX (`mlxir`)
- A phased implementation plan for IR export from MLX
- A phased implementation plan for `mlxir -> ONNX`
- A runtime plan for WebGPU deployment

## 2. Background

MLX currently has:

- Internal computation graph representation based on `array` + `Primitive`
- `.mlxfn` export/import for MLX front-end interoperability
- DOT graph export for visualization

MLX does not currently provide a stable external, tool-friendly IR specifically designed for interoperability with ONNX/WebGPU toolchains.

## 3. Goals

1. Define and implement a stable, versioned MLX external graph IR.
2. Export MLX traced graphs to this IR with deterministic output.
3. Convert this IR to ONNX for a practical subset of operators.
4. Enable browser execution through ONNX runtimes with WebGPU support.
5. Provide clear operator coverage, diagnostics, and fallback behavior.

## 4. Non-Goals (v1)

1. Full parity for every MLX primitive.
2. Training graph export.
3. Automatic export of custom backend kernels to WebGPU.
4. Full dynamic control-flow lowering to ONNX control-flow ops in v1.

## 5. Users and Use Cases

1. Engineers deploying MLX-authored inference models to web apps.
2. Researchers requiring an inspectable external graph format.
3. Tooling developers building conversion, analysis, and optimization tooling.

Primary use cases:

1. Export a traced MLX model to a portable IR artifact.
2. Convert IR to ONNX and validate correctness.
3. Run ONNX in browser with WebGPU, with controlled fallback when required.

## 6. Proposed Architecture

### 6.1 Canonical IR Strategy

Use a **custom MLX Graph IR (`mlxir`)** as the canonical intermediate.

Rationale:

1. MLX primitive semantics exceed direct ONNX coverage in several areas.
2. A dedicated IR allows explicit decomposition/normalization before ONNX lowering.
3. It decouples capture concerns from target runtime concerns.

### 6.2 High-Level Pipeline

1. MLX trace capture -> internal tape
2. Internal tape -> `mlxir` artifact
3. `mlxir` normalization/validation
4. `mlxir` -> ONNX lowering
5. ONNX runtime execution on WebGPU (with optional fallback)

## 7. Functional Requirements

1. Exporter must emit deterministic, versioned IR.
2. IR must represent:
- Inputs/outputs
- Node topology and op names
- Attributes/state
- Constants (inline and/or external blob references)
- Dtype and shape metadata
3. IR export must support multi-trace signatures.
4. ONNX lowering must provide:
- Operator mapping registry
- Decomposition pipeline
- Unsupported-op diagnostics
5. Tooling must provide validation and parity checks.
6. Web runtime plan must include supported/fallback execution strategy.

## 8. Non-Functional Requirements

1. Determinism: repeated export of identical graph should produce stable IR.
2. Observability: clear error messages for unsupported operators/shape issues.
3. Performance: avoid unnecessary graph retracing and duplicate constants.
4. Compatibility: explicit versioning for IR schema and converter.

## 9. Phased Plan

## Phase 0: Scope and Operator Inventory

### Deliverables

1. Primitive inventory from exportable MLX primitives.
2. Coverage classification matrix:
- Direct ONNX mapping
- Decompose needed
- Unsupported/custom
3. v1 scope decision document (inference-only, static-shape-first).

### Exit Criteria

1. Signed-off v1 operator list.
2. Signed-off unsupported operator behavior.

## Phase 1: `mlxir` Schema and Spec

### Deliverables

1. `mlxir` schema specification (JSON schema or protobuf schema + markdown spec).
2. Versioning policy (`ir_version`, producer version, compatibility guarantees).
3. Canonical ordering and naming rules.

### Exit Criteria

1. Schema reviewed and frozen for v1.
2. Reference examples included for single-trace and multi-trace models.

## Phase 2: MLX Core IR Exporter

### Deliverables

1. C++ API to export traced graph as `mlxir`.
2. Implementation reusing existing trace/export pipeline and primitive metadata.
3. Ruby binding support for IR export.
4. File and in-memory export options.

### Exit Criteria

1. Unit tests for export correctness and determinism.
2. Integration tests for representative graphs (single and multi-output ops).

## Phase 3: Validation and Normalization

### Deliverables

1. IR validator:
- DAG integrity
- Type/shape consistency
- Constant/reference integrity
2. Normalization pass set:
- Canonical attrs
- Explicit broadcast forms
- Optional constant folding
3. WebGPU compatibility report tool.

### Exit Criteria

1. Validator catches malformed artifacts with actionable errors.
2. Normalization is deterministic and test-covered.

## Phase 4: `mlxir -> ONNX` Lowering (Core Set)

### Deliverables

1. Converter implementation with op mapping registry.
2. Core op lowering support:
- Elementwise ops
- Cast/reshape/transpose/squeeze/expand
- Matmul/gemm
- Reductions
- Concat/slice/gather/scatter
- Conv
- Softmax and key arg ops
3. Initializer/external-data handling for constants.

### Exit Criteria

1. ONNX checker passes for supported models.
2. Numerical parity suite passes on core models.

## Phase 5: WebGPU Runtime Deployment Path

### Deliverables

1. Runtime harness for ONNX + WebGPU execution in browser.
2. Fallback strategy (`webgpu` primary, CPU/wasm fallback where supported).
3. Benchmark and telemetry report:
- Model load/compile latency
- Steady-state inference latency
- Fallback partition ratio

### Exit Criteria

1. Browser smoke tests pass on target environments.
2. Performance and fallback metrics are published.

## Phase 6: Hard Ops and Custom Extensions

### Deliverables

1. Strategy per hard operator class (e.g., RoPE, RMSNorm, attention variants, quantized ops):
- Decompose to standard ONNX where feasible
- Otherwise custom domain op with runtime constraint annotations
2. Strict/permissive conversion modes.

### Exit Criteria

1. Unsupported-op behavior is explicit and test-covered.
2. Documented compatibility matrix for web deployment.

## 10. Milestones

1. M1: Phase 0+1 complete (spec finalized).
2. M2: Phase 2 complete (exporter functional in C++ and Ruby).
3. M3: Phase 3+4 complete (validated ONNX for core operator set).
4. M4: Phase 5 complete (browser WebGPU path operational).
5. M5: Phase 6 complete (hard-op strategy + compatibility matrix).

## 11. Risks and Mitigations

1. **Operator semantic mismatch (MLX vs ONNX)**
- Mitigation: decomposition layer and explicit mapping tests.
2. **Dynamic-shape and control-flow complexity**
- Mitigation: static-shape-first scope; multi-trace model output strategy.
3. **Runtime fallback unpredictability on web**
- Mitigation: compatibility profiling and fallback metrics in CI.
4. **Version drift across schema/converter/runtime**
- Mitigation: strict versioning and compatibility checks.

## 12. Validation Strategy

1. Unit tests:
- IR emission determinism
- Schema and validator checks
- Op-level lowering correctness
2. Integration tests:
- End-to-end export -> convert -> run parity
- Multi-trace signature selection correctness
3. Runtime tests:
- Browser execution smoke tests
- Fallback behavior checks

## 13. Acceptance Criteria (v1)

1. `mlxir` schema is stable and documented.
2. MLX can export representative inference graphs to `mlxir`.
3. Converter generates valid ONNX for the agreed core op set.
4. Core model suite runs with numerical parity within defined tolerances.
5. WebGPU runtime path is documented, tested, and operational.

## 14. Open Questions

1. Should multi-trace exports produce one ONNX per signature or a dispatcher wrapper?
2. Which quantization patterns are in v1 scope for ONNX lowering?
3. Do we prioritize ONNX standard ops only, or allow custom domains in v1?
4. What minimum browser/runtime matrix is required for GA?

