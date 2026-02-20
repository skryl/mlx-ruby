# Examples Model ONNX/WebGPU Op Coverage RFP

## Status

Completed on February 20, 2026.

Phase progress update (local):

1. Phase 0 baseline coverage task hardening completed and green.
2. Phase 1 (`AsType`, `Sin`, `Cos`, `Erf`, `Less`) implemented and green.
3. Phase 2 (`GatherAxis`, `Pad`, `Equal`, `Floor`) implemented and green.
4. Phase 3 (`LogSumExp`, `Scan/CumSum`) implemented and green.
5. Direct examples ONNX export flow now captures non-empty GraphIR payloads via local eval-time hook and drives the local WebGPU harness path.
6. Capture candidate selection now prefers non-trivial eval graph captures (validated with cvae/segment_anything/encodec regression tests).
7. Examples WebGPU benchmark lane now skips unsupported-op models in non-strict mode (`REQUIRE_WEBGPU=1` remains strict).
8. Coverage task now uses direct examples ONNX capture (`examples/benchmark/benchmark_mlx_examples/onnx_capture_hook.rb`) instead of dot-graph reconstruction for submodule mode.
9. `GatherAxis` lowering now supports broadcast on non-axis dims via `Expand` + `GatherElements` and unblocks `clip`.
10. Phase 4 full-model export parity is green: 23/23 examples models report `unsupported_nodes == 0` with empty `errors_by_model`.
11. Added full examples-submodule ONNX Runtime parity gate (`phase324`) that captures each model, exports ONNX, runs in Python ORT, and checks output parity.
12. `rake benchmark:all` passes across cpu/gpu/webgpu lanes with explicit WebGPU skip policy on submodule examples models when runtime provider is unavailable.
13. CI/report integration completed.
14. Post-completion regression patch applied: `Arange` now accepts numeric arguments emitted as floats, wrapped signed-int64 values are normalized before ONNX initializer emission, and phase313/314/319/323/324 parity gates are green again.
15. Submodule ONNX capture flow now supports CPU-only CI by capturing from dryrun pass (`MLX_BENCHMARK_DRYRUN=1`) and writing fallback payloads at process exit; phase320/321/323/324 gates are green with this path.

## Context

We need GraphIR-to-ONNX export coverage sufficient to run the full model set in `../../mlx-ruby-examples/master` on ONNX Runtime/WebGPU.

A benchmark-path sweep was run across all 23 model scripts declared in `benchmark/runner.rb` and op usage was captured from GraphIR traces.

Coverage result:

1. 23/23 model benchmark scripts were traced.
2. 46 unique GraphIR ops were observed.
3. 11 ops are currently missing for end-to-end coverage of all examples models.

Missing ops:

1. `AsType`
2. `Sin`
3. `Cos`
4. `Erf`
5. `Less`
6. `GatherAxis`
7. `LogSumExp`
8. `Pad`
9. `Equal`
10. `Floor`
11. `Scan` (MLX `cumsum`)

Observed frequency across the 23-model sweep:

1. `Sin` (23)
2. `AsType` (15)
3. `Erf` (9)
4. `Cos` (8)
5. `Less` (7)
6. `GatherAxis` (2)
7. `Pad` / `Equal` / `Floor` / `LogSumExp` / `Scan` (1 each)

## Goal

Implement missing GraphIR lowering and shape inference so all examples benchmark models are exportable to ONNX and runnable for parity checks, with WebGPU compatibility as a first-class target.

## Non-Goals

1. Redesigning model code in `mlx-ruby-examples`.
2. Supporting dynamic-shape/export modes beyond current static benchmark traces.
3. Solving generic ONNX portability issues unrelated to missing op coverage.

## Scope

In scope model set (from `benchmark/runner.rb`):

1. `bert`
2. `cifar`
3. `clip`
4. `cvae`
5. `encodec`
6. `flux`
7. `gcn`
8. `llava`
9. `llms/gguf_llm`
10. `llms/llama`
11. `llms/mistral`
12. `llms/mixtral`
13. `llms/speculative_decoding`
14. `lora`
15. `mnist`
16. `musicgen`
17. `normalizing_flow`
18. `segment_anything`
19. `speechcommands`
20. `stable_diffusion`
21. `t5`
22. `transformer_lm`
23. `whisper`

## Phased Plan (Red/Green)

## Phase 0: Harness + Baseline Gate

### Red

1. Add an examples-model coverage task that exports each model trace to GraphIR and reports unsupported ops.
2. Add failing tests for the known-missing op list above.

### Green

1. Wire a reproducible coverage report artifact (`unsupported_ops_by_model` + union).
2. Lock baseline so subsequent phases can prove coverage closure.

### Exit Criteria

1. Coverage task runs all 23 models and produces deterministic unsupported-op output.
2. Missing-op baseline matches this RFP list.

## Phase 1: High-Frequency Core Ops

### Red

1. Add op-level export/parity tests for `AsType`, `Sin`, `Cos`, `Erf`, `Less`.
2. Add targeted model fixtures that currently fail due to those ops.

### Green

1. Implement lowering:
   - `AsType` -> ONNX `Cast`
   - `Sin` -> ONNX `Sin`
   - `Cos` -> ONNX `Cos`
   - `Erf` -> ONNX `Erf`
   - `Less` -> ONNX `Less`
2. Add shape inference updates where needed for elementwise compare/cast paths.

### Exit Criteria

1. New op-level tests pass.
2. Unsupported-op count drops for all impacted models.

## Phase 2: Indexing + Padding + Scalar Compare

### Red

1. Add failing tests for `GatherAxis`, `Pad`, `Equal`, `Floor`.
2. Add axis/shape edge-case tests for `GatherAxis` and padding mode/value tests for `Pad`.

### Green

1. Implement lowering:
   - `GatherAxis` -> ONNX `GatherElements` (axis-aware)
   - `Pad` -> ONNX `Pad` (constant mode first)
   - `Equal` -> ONNX `Equal`
   - `Floor` -> ONNX `Floor`
2. Extend shape inference for gather-elements/pad paths.

### Exit Criteria

1. Op-level tests pass.
2. Models blocked only by `LogSumExp`/`Scan` are isolated.

## Phase 3: Reduction/Scan Completion

### Red

1. Add failing tests for `LogSumExp` and `Scan` (`cumsum` semantics).
2. Add axis/keepdims/exclusive/reverse coverage tests for `Scan`.

### Green

1. Implement lowering:
   - `LogSumExp` -> ONNX `ReduceLogSumExp` (or numerically stable decomposition)
   - `Scan` (`cumsum`) -> ONNX `CumSum`
2. Add shape inference updates for both ops.

### Exit Criteria

1. Op-level tests pass.
2. No remaining unsupported ops for the 23-model sweep.

## Phase 4: Full Model Export + Runtime Parity

### Red

1. Add all-model export tests requiring `unsupported_nodes == 0`.
2. Add ONNX Runtime CPU parity tests for all 23 models.
3. Add WebGPU parity lane for models runnable in browser harness constraints.

### Green

1. Resolve regressions exposed by full-model tests.
2. Stabilize thresholds and deterministic seeds.
3. Publish final coverage and parity report.

### Exit Criteria

1. 23/23 models export with `unsupported_nodes == 0`.
2. ONNX Runtime CPU parity passes for all models.
3. WebGPU parity passes for supported harness subset with explicit skip policy documented.

## Acceptance Criteria

1. Missing-op list in this RFP is fully implemented.
2. `graph_ir_webgpu_compatibility_report` returns zero unsupported nodes for every benchmark model trace.
3. ONNX Runtime parity checks pass within agreed tolerances for all benchmark models.
4. New coverage tasks are wired into CI or a required pre-merge benchmark lane.

## Deliverables

1. GraphIR lowering + shape inference implementations for all missing ops.
2. Op-level parity tests (stub conversion + ORT runtime).
3. Examples-model coverage task and report artifact.
4. Full-model parity test suite for `mlx-ruby-examples` benchmark model set.
5. Documentation update with supported op matrix and known caveats.

## Risks And Mitigations

1. Risk: Semantic mismatch for `GatherAxis` and `CumSum` flags.
   - Mitigation: Explicit axis/exclusive/reverse fixture tests and per-op parity checks.
2. Risk: Numerical sensitivity in `LogSumExp` and mixed precision (`AsType`).
   - Mitigation: Stable lowering choice and model-specific tolerance envelopes.
3. Risk: Browser WebGPU variability.
   - Mitigation: Strict provider selection, deterministic inputs, skip-with-reason policy when unavailable.

## Implementation Checklist

- [x] Phase 0 baseline coverage task and failing-op fixtures added.
- [x] Phase 1 (`AsType`, `Sin`, `Cos`, `Erf`, `Less`) implemented and green.
- [x] Phase 2 (`GatherAxis`, `Pad`, `Equal`, `Floor`) implemented and green.
- [x] Phase 3 (`LogSumExp`, `Scan/CumSum`) implemented and green.
- [x] Direct examples ONNX export capture hook implemented with non-empty-graph regression coverage.
- [x] Capture selection regressions covered for cvae/segment_anything/encodec benchmark scripts.
- [x] Examples WebGPU unsupported-op handling defaults to skip (strict mode via `REQUIRE_WEBGPU=1`).
- [x] Phase 4 full 23-model export/parity suites green.
- [x] Phase 4 all-model Python ONNX Runtime parity suite green.
- [x] CI lane/report integration completed.
- [x] Post-completion regression fix for numeric argument normalization (`Arange` float args and wrapped int64 sentinel values) validated by phase313/314/319/323/324 gates.
- [x] Post-completion CI hardening for submodule capture on CPU-only runners (`phase320/321/323/324`).
