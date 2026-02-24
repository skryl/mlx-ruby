# Python ONNX Unit Superset Migration PRD

Status: Completed (2026-02-24)

## Context

`submodules/mlx-onnx/python/tests/unit` currently has almost no source tests, while
`submodules/mlx-onnx/python/tests/test_onnx.py` holds core API/export coverage and
`test/reports/unit_onnx_test_inventory_tmp.md` documents 133 Ruby ONNX unit test methods
across 49 files. The requested target is:

1. Python-only tests under `submodules/mlx-onnx/python/tests/unit`.
2. Op-specific unit files.
3. Coverage that is a strict superset of the Ruby ONNX unit inventory.
4. No legacy/old API checks and no webgpu harness/browser/benchmark/docs migration in this effort.

## Goals

1. Establish per-op Python unit tests in `python/tests/unit`.
2. Migrate all applicable coverage from `python/tests/test_onnx.py` into unit files.
3. Add a machine-checkable superset gate against the Ruby inventory document.
4. Keep tests deterministic and runnable as targeted subsets.

## Non-Goals

1. Exporter implementation changes.
2. Browser/webgpu harness, benchmark, or docs drift checks.
3. Reintroducing legacy API surface assertions.
4. Reorganizing integration tests beyond what unit extraction requires.

## Phased Plan (Red/Green/Refactor)

### Phase 1: Unit Harness + Superset Inventory Gate

Red:

1. Add a failing inventory superset guard test in `python/tests/unit`.
2. Add inventory artifact input under `python/tests/unit/_support` from
   `test/reports/unit_onnx_test_inventory_tmp.md`.

Green:

1. Add shared helpers (`conftest.py` and `_support/*`) for deterministic paths,
   optional dependency guards, and common ONNX assertions.
2. Add a maintained mapping artifact that tracks implemented Python unit coverage.
3. Make the superset guard pass by validating mapping completeness/shape.

Refactor:

1. Normalize helper boundaries so op test files stay thin.

Exit Criteria:

1. `python/tests/unit` has reusable scaffolding.
2. Superset gate exists and passes locally.

### Phase 2: Migrate Core API/Export Coverage Out of `test_onnx.py`

Red:

1. Add failing unit tests per API area (`export_ir*`, `ir_to_onnx*`, direct export parity,
   external data, compatibility reports, runtime parity, oracle parity, shapeless).

Green:

1. Port tests into focused unit files:
   - `test_export_ir_contract.py`
   - `test_ir_to_onnx_contract.py`
   - `test_export_onnx_binary_contract.py`
   - `test_export_onnx_external_data.py`
   - `test_compatibility_reports.py`
   - `test_runtime_parity.py`
   - `test_python_oracle_parity.py`
2. Keep shared setup/utilities in `_support`.

Refactor:

1. Remove duplicated helper code from test files.

Exit Criteria:

1. Equivalent coverage for the migrated areas exists in `python/tests/unit`.
2. `python/tests/test_onnx.py` is reduced or deleted for migrated scopes.

### Phase 3: Add Op-Specific Tests as Ruby-Inventory Superset

Red:

1. Add failing per-op files for inventory-listed op groups.
2. Add missing-case checks for inventory methods not yet represented.

Green:

1. Create op-specific files (one op/op-group per file) covering lowering, contract,
   and runtime assertions as applicable.
2. Ensure every Ruby inventory method has at least one mapped Python unit test.
3. Add Python-only extensions beyond Ruby coverage for native-only behavior where relevant.

Refactor:

1. Group heavy fixtures and payload builders into reusable support helpers.

Exit Criteria:

1. Ruby inventory superset gate passes.
2. Op-specific unit suite is discoverable and stable.

### Phase 4: Decommission Legacy Layout + Final Gates

Red:

1. Remove obsolete monolithic files and confirm missing-import/discovery failures.

Green:

1. Delete or fully retire `python/tests/test_onnx.py` (for migrated scopes).
2. Run targeted and broader gates:
   - `python -m pytest -q python/tests/unit`
   - `python -m pytest -q python/tests/test_onnx.py` (only if residual tests remain)
   - `python -m pytest -q python/tests/integration/test_examples.py` (smoke check)

Refactor:

1. Final naming cleanup for consistency (`test_op_*`, `test_export_*`, `test_contract_*`).

Exit Criteria:

1. New unit layout is the primary test entrypoint.
2. Required gates pass.
3. PRD status can be set to Completed.

## Acceptance Criteria

1. `python/tests/unit` contains op-specific tests and shared support utilities.
2. Ruby inventory coverage is represented as a machine-checkable superset in Python tests.
3. `test_onnx.py` no longer carries migrated unit scopes.
4. Targeted and broad gates for touched areas are green.

## Risks and Mitigations

1. Risk: Coverage drift between Ruby inventory and Python mapping.
   - Mitigation: keep inventory artifact and guard test in CI path.
2. Risk: Runtime-dependent tests become flaky across environments.
   - Mitigation: explicit dependency guards + deterministic input fixtures.
3. Risk: Excessive duplication across op tests.
   - Mitigation: shared `_support` fixtures/builders and strict thin-file pattern.

## Implementation Checklist

- [x] Phase 1 Red completed.
- [x] Phase 1 Green completed.
- [x] Phase 1 Refactor completed.
- [x] Phase 2 Red completed.
- [x] Phase 2 Green completed.
- [x] Phase 2 Refactor completed.
- [x] Phase 3 Red completed.
- [x] Phase 3 Green completed.
- [x] Phase 3 Refactor completed.
- [x] Phase 4 Red completed.
- [x] Phase 4 Green completed.
- [x] Phase 4 Refactor completed.

## Execution Log

1. 2026-02-24: PRD updated to include Ruby inventory superset gate and `test_onnx.py` decomposition plan.
2. 2026-02-24: Phase 1 implemented in `python/tests/unit` with shared support modules, inventory mapping artifact (133 entries), and gate test.
3. 2026-02-24: Targeted tests passed: `python -m pytest -q python/tests/unit/test_inventory_superset_gate.py python/tests/unit/test_export_ir_contract.py` (6 passed).
4. 2026-02-24: Phase 2 started by migrating `export_ir`, `ir_to_onnx`, and binary export contract checks into dedicated `python/tests/unit` files.
5. 2026-02-24: Updated inventory mapping to 17 implemented Ruby inventory entries and reran targeted tests: `python -m pytest -q python/tests/unit/test_inventory_superset_gate.py python/tests/unit/test_export_onnx_binary_contract.py python/tests/unit/test_ir_to_onnx_contract.py python/tests/unit/test_export_ir_contract.py` (22 passed).
6. 2026-02-24: Added unit files for external-data contracts, compatibility reports, shapeless export contract, and runtime parity.
7. 2026-02-24: Additional targeted gates passed:
   - `python -m pytest -q python/tests/unit/test_export_onnx_external_data_contract.py python/tests/unit/test_compatibility_reports_contract.py python/tests/unit/test_shapeless_contract.py` (6 passed).
   - `python -m pytest -q python/tests/unit/test_runtime_parity_contract.py` (2 passed).
   - `python -m pytest -q python/tests/unit/test_inventory_superset_gate.py` (2 passed).
8. 2026-02-24: Ruby inventory mapping now marks 21/133 entries as implemented.
9. 2026-02-24: Added Python oracle parity migration (`test_python_oracle_parity.py`) and moved ONNX python-oracle builder/model-summary logic into shared `_support/oracle.py`.
10. 2026-02-24: Added Ruby parity bridge harness (`_support/ruby_parity.py`) and generated 49 Python parity files (`test_ruby_parity_*.py`) covering all inventory Ruby files.
11. 2026-02-24: Updated `ruby_inventory_mapping.json` to 133/133 implemented entries and tightened gate to require all entries implemented.
12. 2026-02-24: Retired monolithic `python/tests/test_onnx.py` after migrating its coverage.
13. 2026-02-24: Registered pytest marker `ruby_parity` in `submodules/mlx-onnx/pyproject.toml`.
14. 2026-02-24: Final test gates passed:
    - `python -m pytest -q python/tests/unit` (32 passed, 49 skipped)
    - `python -m pytest -q python/tests/integration/test_examples.py` (20 passed)
    - `python -m pytest -q python/tests` (52 passed, 49 skipped)
    - `MLX_ONNX_RUN_RUBY_PARITY=1 python -m pytest -q python/tests/unit/test_ruby_parity_argreduce_lowering.py -k test_argmax_axis1_lowers_argreduce_to_argmax_plus_cast` (1 passed, parity bridge smoke)
15. 2026-02-24: Added regeneration utility `python/tests/unit/_support/generate_ruby_parity_tests.py` and documented operational mode: Ruby parity wrappers are opt-in and skip by default unless `MLX_ONNX_RUN_RUBY_PARITY=1`.
16. 2026-02-24: Executed full Ruby parity bridge suite:
    - `MLX_ONNX_RUN_RUBY_PARITY=1 python -m pytest -q python/tests/unit/test_ruby_parity_*.py` (133 passed).
17. 2026-02-24: Executed full Python tests with parity enabled:
    - `MLX_ONNX_RUN_RUBY_PARITY=1 python -m pytest -q python/tests` (185 passed).
18. 2026-02-24: Corrective pass: removed Ruby-invoking unit wrappers (`test_ruby_parity_*.py`) and removed Ruby bridge/generator support files from `python/tests/unit/_support`.
19. 2026-02-24: Replaced wrappers with Python-native per-op inventory files (`test_<inventory_stem>.py`) that execute `mlx_onnx` checks via `_support/inventory_native_runner.py` (no Ruby invocation).
20. 2026-02-24: Updated `ruby_inventory_mapping.json` to point all 133 inventory entries at Python-native unit files (`python/tests/unit/test_<stem>.py::test_inventory_case[...]`).
21. 2026-02-24: Removed now-obsolete `ruby_parity` marker from `submodules/mlx-onnx/pyproject.toml`.
22. 2026-02-24: Final corrected gates:
    - `python -m pytest -q python/tests/unit` (165 passed)
    - `python -m pytest -q python/tests` (185 passed)
