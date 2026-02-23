# Python IR Test Suite Expansion PRD

Status: Completed (2026-02-23)

## Context

`mlx/python/tests/test_ir.py` currently has minimal coverage (3 tests) for the native IR/ONNX surface. The Ruby-side IR suite covers substantially more API contract, validation, binary parity, external-data behavior, and oracle/runtime paths.

We need to expand the Python IR suite with the agreed test set while explicitly excluding:

1. Old/legacy method surface checks.
2. WebGPU harness/browser tests.
3. Benchmark/docs tests.

## Goals

1. Add all agreed tests to `mlx/python/tests/test_ir.py` (24 test methods).
2. Cover contract, serialization, validation, binary parity, external data, runtime parity, oracle parity, and shapeless behavior.
3. Keep tests deterministic and dependency-gated where external Python packages are required.

## Non-Goals

1. Adding or asserting legacy/removed method aliases.
2. Porting WebGPU harness/browser/coverage tests.
3. Porting benchmark/docs drift tests.
4. Changing runtime/export implementation behavior as part of this PRD.

## Public API/Interface Impact

No API changes. Tests exercise existing Python `mlx.core` IR methods:

1. `export_ir`
2. `export_ir_json`
3. `export_onnx_compatibility_report`
4. `export_onnx_json`
5. `export_onnx`
6. `ir_to_onnx_json`
7. `ir_to_onnx`
8. `ir_compatibility_report_json`

## Test Inventory (Must Add)

1. `test_export_ir_json_matches_export_ir`
2. `test_export_ir_is_deterministic`
3. `test_export_ir_includes_keyword_inputs`
4. `test_export_ir_includes_constant_values`
5. `test_ir_to_onnx_json_accepts_json_string_source`
6. `test_ir_to_onnx_json_accepts_path_source`
7. `test_ir_to_onnx_json_rejects_malformed_json`
8. `test_ir_to_onnx_json_rejects_malformed_payload`
9. `test_ir_to_onnx_json_rejects_malformed_constants`
10. `test_ir_to_onnx_json_rejects_unknown_op_as_not_implemented`
11. `test_ir_to_onnx_binary_rejects_constant_value_count_mismatch`
12. `test_export_onnx_rejects_non_path_target`
13. `test_ir_to_onnx_rejects_non_path_target`
14. `test_export_onnx_direct_matches_ir_to_onnx_binary`
15. `test_export_onnx_external_data_writes_sidecar`
16. `test_export_onnx_external_data_uses_custom_filename`
17. `test_ir_to_onnx_external_data_writes_sidecar`
18. `test_export_onnx_compatibility_report_lists_unsupported_ops`
19. `test_ir_compatibility_report_json_matches_dict_report`
20. `test_onnx_runtime_parity_exp_add`
21. `test_onnx_runtime_parity_with_initializer_constants`
22. `test_native_binary_matches_python_onnx_oracle_no_external_data`
23. `test_native_binary_matches_python_onnx_oracle_with_external_data`
24. `test_shapeless_export_ir_and_onnx_json`

## Phased Plan (Red/Green)

### Phase 1: Contract + Serialization Baseline

Red:

1. Add failing tests for inventory items 1-6.
2. Capture baseline failures for payload equality, determinism, keyword mapping, constants, and source type handling.

Green:

1. Implement shared test helpers in `test_ir.py` for:
   - temporary file management
   - common sample function/input creation
   - IR payload export/parse wrappers
2. Make inventory items 1-6 pass.

Exit Criteria:

1. Items 1-6 pass locally.
2. No flakiness in determinism assertion (`export_ir_json` exact string match).

### Phase 2: Validation + Unsupported/Error Semantics

Red:

1. Add failing tests for inventory items 7-11 and 18-19.
2. Capture expected error class/message behavior for malformed source/payload/constants and unsupported ops.

Green:

1. Implement validation-path tests with explicit assertions on:
   - error class (`ValueError`/`RuntimeError`/`NotImplementedError`, as bound)
   - message tags (`[ir.api]` and unsupported lowering semantics)
2. Ensure compatibility report parity checks (dict vs JSON-source path) pass.

Exit Criteria:

1. Items 7-11, 18-19 pass.
2. Unsupported-op path is asserted both in report and conversion failure.

### Phase 3: Binary + External Data + Direct Parity

Red:

1. Add failing tests for inventory items 12-17.
2. Capture baseline binary mismatch or external-data emission failures.

Green:

1. Add binary helpers in `test_ir.py` for file read/size assertions and temporary artifact creation.
2. Validate:
   - target path validation behavior
   - direct export vs IR-source binary parity (byte equality)
   - external data sidecar creation and custom filename behavior
3. Use deterministic model_name/opset inputs to avoid output drift.

Exit Criteria:

1. Items 12-17 pass.
2. No intermediate/public Python ONNX builder path is used for native export assertions.

### Phase 4: ONNX Runtime Parity

Red:

1. Add failing tests for inventory items 20-21.
2. Skip with explicit message when `onnxruntime` is unavailable.

Green:

1. Add embedded Python runner snippets (or helper functions) to execute ONNX with typed feeds.
2. Assert numeric parity for:
   - exp+add graph
   - initializer-constant graph

Exit Criteria:

1. Items 20-21 pass when dependencies exist.
2. Tests skip cleanly with actionable messages when dependencies are absent.

### Phase 5: Native-vs-Python ONNX Oracle

Red:

1. Add failing tests for inventory items 22-23.
2. Capture baseline mismatch signal between native binary and Python ONNX oracle summary.

Green:

1. Add Python-oracle builder and model-summary helper logic to `test_ir.py` (adapted for Python test style).
2. Compare normalized model summaries:
   - no external data
   - external data enabled

Exit Criteria:

1. Items 22-23 pass.
2. Oracle summary comparison is deterministic.

### Phase 6: Shapeless Coverage

Red:

1. Add failing test for inventory item 24.

Green:

1. Add shapeless test path for IR export + ONNX JSON conversion with assertions on payload/stub validity.

Exit Criteria:

1. Item 24 passes.
2. Shapeless path is validated without introducing legacy method checks.

## Test Execution Gates

1. Targeted:
   - `cd mlx && DEVICE=cpu python -m unittest -v python.tests.test_ir.TestGraphIr`
2. File-level:
   - `cd mlx && DEVICE=cpu python -m unittest discover -v python/tests -p 'test_ir.py'`
3. Immediate regression:
   - `cd mlx && DEVICE=cpu python -m unittest discover -v python/tests -p 'test_export_import.py'`

If dependencies are missing:

1. Document which tests were skipped due to missing `onnx` or `onnxruntime`.
2. Keep non-dependency tests green.

## Execution Log

1. Targeted:
   - `cd mlx && DEVICE=cpu PYTHONPATH=python/tests python -m unittest -v test_ir.TestGraphIr`
   - Result: `Ran 24 tests ... OK`
2. File-level:
   - `cd mlx && DEVICE=cpu python -m unittest discover -v python/tests -p 'test_ir.py'`
   - Result: `Ran 24 tests ... OK`
3. Immediate regression:
   - `cd mlx && DEVICE=cpu python -m unittest discover -v python/tests -p 'test_export_import.py'`
   - Result: `Ran 16 tests ... OK`

## Acceptance Criteria

1. All 24 inventory tests are implemented in `mlx/python/tests/test_ir.py`.
2. No added tests check legacy/removed methods.
3. No added tests cover WebGPU harness/browser/benchmark/docs.
4. Contract/validation/binary/external-data/runtime/oracle/shapeless paths are each covered by at least one test.
5. `test_ir.py` passes on CPU with dependency-based skips only where expected.
6. `test_export_import.py` remains green after the expansion.

## Risks and Mitigations

1. Risk: Optional dependency variance (`onnx`, `onnxruntime`) causes CI/local mismatch.
   - Mitigation: explicit dependency checks and skip messages in runtime/oracle tests.
2. Risk: Binary parity assertions become flaky due to nondeterministic metadata.
   - Mitigation: fixed opset/model names and normalized summary comparisons for oracle checks.
3. Risk: Test file grows too large and hard to maintain.
   - Mitigation: add internal helper functions within `test_ir.py` for setup/build/assertion reuse.
4. Risk: Runtime tests increase duration.
   - Mitigation: keep fixtures minimal and reuse shared sample graphs.

## Implementation Checklist

- [x] Phase 1 Red: add failing tests for items 1-6.
- [x] Phase 1 Green: add helpers and pass items 1-6.
- [x] Phase 2 Red: add failing tests for items 7-11 and 18-19.
- [x] Phase 2 Green: pass validation/unsupported/compat-report tests.
- [x] Phase 3 Red: add failing tests for items 12-17.
- [x] Phase 3 Green: pass binary/external-data/direct parity tests.
- [x] Phase 4 Red: add failing tests for items 20-21.
- [x] Phase 4 Green: pass ONNX runtime parity tests (or documented skips).
- [x] Phase 5 Red: add failing tests for items 22-23.
- [x] Phase 5 Green: pass native-vs-python oracle parity tests.
- [x] Phase 6 Red: add failing test for item 24.
- [x] Phase 6 Green: pass shapeless path test.
- [x] Run targeted + file-level + regression gates and record outcomes.
- [x] Mark PRD status updated from Proposed -> In Progress -> Completed when done.
