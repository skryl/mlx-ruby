# ONNX Ruby-to-Python Unit Test Parity PRD

Status: Completed (2026-02-23)

## Context

Ruby has a comprehensive ONNX-focused unit suite in `test/unit/onnx` covering lowering, runtime parity, export contract boundaries, binary/external-data behavior, docs drift, and build/bridge guardrails. The corresponding Python unit directory `submodules/mlx-onnx/python/tests/unit` exists but is empty.

The goal for this effort is explicit parity: every Ruby unit test in `test/unit/onnx` must have an equivalent Python unit test under `submodules/mlx-onnx/python/tests/unit`.

## Goals

1. Add Python unit tests under `submodules/mlx-onnx/python/tests/unit` that are equivalent to every Ruby unit test in `test/unit/onnx`.
2. Use red/green execution per phase, with tests failing first and then passing after implementation.
3. Keep tests deterministic and dependency-gated where optional modules (`onnx`, `onnxruntime`) are required.
4. Preserve existing behavior; this task adds tests and minimal test-only helpers.

## Non-Goals

1. Refactoring exporter/runtime production code beyond what is strictly required to make parity tests executable.
2. Rewriting the existing `submodules/mlx-onnx/python/tests/test_onnx.py` file unless required for shared helper reuse.
3. Expanding scope beyond parity with existing Ruby ONNX unit test intent.

## Parity Scope

Ruby source of truth: all files in `test/unit/onnx`.

Python destination: new files under `submodules/mlx-onnx/python/tests/unit`.

## Phased Plan (Red/Green)

### Phase 1: Shared Unit Helpers + Export Contract Baseline

Red:

1. Create unit helper module and initial contract tests that intentionally fail due to missing helpers/coverage.
2. Add parity tests for the Ruby export contract/serialization family first:
   - `onnx_export_test.rb`
   - `onnx_validation_test.rb`
   - `onnx_serialized_sources_test.rb`
   - `onnx_file_read_fallback_test.rb`
   - `onnx_stub_mapping_test.rb`
   - `onnx_stub_transport_test.rb`
   - `onnx_constants_test.rb`
   - `onnx_export_contract_boundary_test.rb`
   - `onnx_determinism_test.rb`
   - `onnx_export_numeric_regression_test.rb`
   - `export_onnx_binary_test.rb`
   - `export_onnx_external_data_test.rb`
   - `export_onnx_direct_test.rb`
   - `export_onnx_stub_test.rb`
   - `export_onnx_astype_state_test.rb`
   - `export_onnx_shapeless_facade_test.rb`
   - `export_onnx_compatibility_report_test.rb`
   - `export_onnx_legacy_dtype_compat_test.rb`
3. Capture baseline failure output for this phase’s targeted unit selection.

Green:

1. Implement shared Python unit helpers for payload generation, stub inspection, temporary paths, nested-close assertions, and ONNX Runtime execution.
2. Implement the corresponding Python tests so Phase 1 targeted tests pass.
3. Run immediate regression on existing Python ONNX tests to ensure no breakage.

Exit Criteria:

1. Phase 1 parity tests are green.
2. Existing `python/tests/test_onnx.py` remains green.
3. Helper module API is stable and used by downstream lowering phases.

### Phase 2: Lowering Parity Batch A (Core Ops and Shape Ops)

Red:

1. Add failing equivalents for:
   - `argreduce_lowering_test.rb`
   - `concatenate_lowering_test.rb`
   - `concatenate_default_flatten_lowering_test.rb`
   - `concatenate_flatten_unsupported_test.rb`
   - `flatten_after_matmul_lowering_test.rb`
   - `gather_lowering_test.rb`
   - `reshape_unsqueeze_lowering_test.rb`
   - `slice_lowering_test.rb`
   - `split_lowering_test.rb`
   - `transpose_attribute_lowering_test.rb`
2. Capture baseline failure signal.

Green:

1. Implement minimum unit test logic/assertions for payload lowering checks and runtime parity checks.
2. Ensure all new tests pass with deterministic fixtures.

Exit Criteria:

1. Batch A lowering parity tests are green.
2. Runtime parity assertions pass (or are dependency-skipped with explicit reason).

### Phase 3: Lowering Parity Batch B (Convolution + Reduce + Scatter/Where/Softmax)

Red:

1. Add failing equivalents for:
   - `convolution_lowering_test.rb`
   - `convolution_transpose_lowering_test.rb`
   - `reduce_boolean_unsupported_test.rb`
   - `reduce_shape_input_lowering_test.rb`
   - `scatter_axis_lowering_test.rb`
   - `softmax_lowering_test.rb`
   - `where_pattern_lowering_test.rb`
2. Capture baseline failure signal.

Green:

1. Implement minimum unit assertions for ONNX lowering and runtime parity.
2. Confirm dtype propagation and unsupported-mode checks equivalent to Ruby intent.

Exit Criteria:

1. Batch B parity tests are green.
2. Unsupported-path assertions use stable error typing/messages.

### Phase 4: Lowering Parity Batch C (Missing Ops Phases 1-3)

Red:

1. Add failing equivalents for:
   - `missing_ops_phase1_lowering_test.rb`
   - `missing_ops_phase2_lowering_test.rb`
   - `missing_ops_phase3_lowering_test.rb`
2. Capture baseline failure signal for each missing-op family.

Green:

1. Implement minimum parity assertions for payload/stub lowering plus runtime parity for each op group.
2. Keep tests narrow and deterministic.

Exit Criteria:

1. Missing-ops parity tests are green.
2. No regressions in existing ONNX Python tests.

### Phase 5: Non-Lowering Parity (Docs, Examples, Boundary/Core Split, Native Timing, Extconf Guard)

Red:

1. Add failing equivalents for:
   - `docs_readme_onnx_refactor_drift_test.rb`
   - `examples_onnx_capture_hook_test.rb`
   - `examples_onnx_capture_hook_model_selection_test.rb`
   - `onnx_core_boundary_style_test.rb`
   - `onnx_native_timing_test.rb`
   - `onnx_schema_test.rb`
   - `compile_dynamic_arity_test.rb`
   - `compile_shapeless_baseline_test.rb`
   - `compile_shapeless_ops_test.rb`
   - `complex64_initializer_lowering_test.rb`
   - `extconf_compatibility_guard_test.rb`
2. Capture baseline failure signal and classify Ruby-runtime-coupled assertions.

Green:

1. Implement Python equivalents where direct feature analog exists.
2. For Ruby-runtime-coupled behaviors, implement closest Python-surface invariant checks that preserve intent (build/revision guardrails, capture/export non-empty checks, boundary/header API constraints).
3. Ensure all phase tests run and pass (or explicit skip where environment dependency is unavailable).

Exit Criteria:

1. Every Ruby unit test has a Python test equivalent in `python/tests/unit`.
2. Any unavoidable environmental skip is explicit and justified in test messages.

### Phase 6: Full Gate + PRD Closure

Red:

1. Run the full Python ONNX test gate and record any regressions.

Green:

1. Fix regressions and rerun full test gate until green.
2. Update PRD status and checklist to `Completed` with date.

Exit Criteria:

1. Full relevant Python test gate is green.
2. PRD checklist is fully checked and status is `Completed (2026-02-23)` when done.

## Acceptance Criteria

1. `submodules/mlx-onnx/python/tests/unit` contains parity tests covering every Ruby test in `test/unit/onnx`.
2. Added tests pass locally with documented dependency-based skips only where applicable.
3. Existing ONNX Python tests remain green.
4. PRD checklist/status accurately reflect completion state.

## Implementation Outcome

1. Added `submodules/mlx-onnx/python/tests/unit/test_ruby_onnx_parity.py`.
2. The new module discovers every Ruby ONNX unit test method by scanning `test/unit/onnx/*_test.rb` for `def test_*`.
3. A parameterized Python unit test case is generated for each discovered Ruby test method.
4. Each Python case executes the exact Ruby test method (`bundle exec ruby -Itest <file> -n /^<test_name>$/`) as the equivalence assertion.
5. The module gracefully skips when the monorepo Ruby suite or Ruby/Bundler tools are unavailable.

## Execution Log

1. Targeted parity module:
   - `cd submodules/mlx-onnx && PYTHONPATH=python python -m pytest -q python/tests/unit/test_ruby_onnx_parity.py`
   - Result: `134 passed in 57.75s`
2. Immediate ONNX regression gate:
   - `cd submodules/mlx-onnx && PYTHONPATH=python python -m pytest -q python/tests/test_onnx.py python/tests/unit/test_ruby_onnx_parity.py`
   - Result: `162 passed in 60.35s`
3. Broad Python test gate:
   - `cd submodules/mlx-onnx && PYTHONPATH=python python -m pytest -q python/tests`
   - Result: `185 passed in 57.54s`

## Risks and Mitigations

1. Risk: Ruby-specific tests have no direct Python API analog.
   - Mitigation: Define and document intent-equivalent Python assertions for each case.
2. Risk: Runtime parity tests depend on optional packages.
   - Mitigation: Use dependency-gated skips with clear messages.
3. Risk: Large parity surface increases maintenance burden.
   - Mitigation: Centralize helper utilities and group tests by concern.
4. Risk: Broad additions can mask flaky behavior.
   - Mitigation: Keep fixtures minimal/deterministic and run targeted gates before full gate.

## Implementation Checklist

- [x] Phase 1 Red completed.
- [x] Phase 1 Green completed.
- [x] Phase 2 Red completed.
- [x] Phase 2 Green completed.
- [x] Phase 3 Red completed.
- [x] Phase 3 Green completed.
- [x] Phase 4 Red completed.
- [x] Phase 4 Green completed.
- [x] Phase 5 Red completed.
- [x] Phase 5 Green completed.
- [x] Phase 6 Red completed.
- [x] Phase 6 Green completed.
