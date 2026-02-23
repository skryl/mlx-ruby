# 2026_02_23 MLX ONNX Split Cleanup PRD

Status: Completed (2026-02-23)

## Context

`mlx-onnx` is now the standalone IR/ONNX library, but cleanup gaps remain in this repo:

1. Git submodule metadata and gitlink index entries are inconsistent (`.gitmodules` points to `submodules/*` while index still had legacy top-level gitlinks).
2. Some docs still reference outdated API method names (`export_ir`, `ir_to_onnx`) instead of current public names.
3. One docs reference still points at the old native file path.
4. Internal C++/Ruby-binding symbol names are still `ir_*`-prefixed even though ownership is now ONNX-focused.
5. Build-time compatibility checks between `submodules/mlx` and `submodules/mlx-onnx` are weak.
6. CI does not explicitly gate submodule coherence + ONNX-native smoke checks.

## Goals

1. Make submodule state canonical and reproducible.
2. Make docs consistent with the current `MLX::ONNX` public interface.
3. Rename internal native binding symbols for clarity without changing public method names.
4. Add an explicit compatibility guard in `ext/mlx/extconf.rb`.
5. Add CI steps that fail early on submodule/ONNX integration drift.

## Non-goals

1. Changing public `MLX::ONNX` method names.
2. Altering ONNX lowering semantics or output format.
3. Reworking broad historical PRDs beyond this targeted cleanup.

## Phased Plan (Red/Green)

### Phase 1: Submodule Canonicalization

Red:
1. `git submodule status` fails due stale top-level gitlinks not matching `.gitmodules`.

Green:
1. Remove legacy top-level gitlinks (`mlx`, `mlx-ruby-examples`) from index.
2. Stage gitlinks for `submodules/mlx`, `submodules/mlx-ruby-examples`, and `submodules/mlx-onnx`.
3. Verify `git submodule status` succeeds.

Exit criteria:
1. Only `submodules/*` submodule gitlinks remain in index.
2. `git submodule status` runs without mapping errors.

### Phase 2: Docs API/Path Alignment

Red:
1. Docs/README include outdated method names or old source-path references.

Green:
1. Replace outdated `MLX::ONNX.export_ir*` / `MLX::ONNX.ir_to_onnx*` references with `export_graph_ir*` / `graph_ir_to_onnx*`.
2. Replace stale docs file path references (e.g., old native source path) with current `ext/mlx-onnx/native.cpp`.
3. Run targeted drift/docs tests.

Exit criteria:
1. No outdated public API names remain outside intentional historical docs.
2. Docs tests covering ONNX API references pass.

### Phase 3: Internal Native Symbol Cleanup

Red:
1. Internal symbol names remain `ir_*` in binding registration path.

Green:
1. Rename internal/non-public C++ symbols and init entrypoint names to ONNX-oriented names.
2. Keep Ruby-visible method strings unchanged (`export_graph_ir`, `graph_ir_to_onnx`, etc.).
3. Rebuild extension and run targeted ONNX tests.

Exit criteria:
1. Internal naming is consistent (`onnx_*`), and public API remains unchanged.
2. ONNX contract tests remain green.

### Phase 4: Build Compatibility Guard

Red:
1. `extconf.rb` does not fail early for mlx/mlx-onnx revision mismatch.

Green:
1. Add compatibility check in `ext/mlx/extconf.rb` comparing the workspace `submodules/mlx` revision against the mlx revision pinned by `submodules/mlx-onnx` (when available).
2. Fail fast with actionable error text on mismatch.
3. Add targeted regression test(s) for guard presence/behavior.

Exit criteria:
1. Guard executes before native build proceeds.
2. Test coverage exists for the guard logic.

### Phase 5: CI Guardrails

Red:
1. CI lacks explicit steps for submodule coherence and ONNX-native smoke checks.

Green:
1. Add CI steps to validate submodule status from clean checkout.
2. Add targeted ONNX-native smoke test command(s) after native build.
3. Ensure release workflow includes equivalent protection.

Exit criteria:
1. Main CI and release CI both enforce submodule + ONNX smoke gates.

## Test Execution Gates

1. `git submodule status`
2. `bundle exec ruby -Itest test/onnx/onnx_validation_test.rb`
3. `bundle exec ruby -Itest test/onnx/onnx_export_contract_boundary_test.rb`
4. `bundle exec ruby -Itest test/onnx/export_onnx_shapeless_facade_test.rb`
5. Targeted docs/drift and extconf guard tests for touched files

Execution results:

1. `git submodule status --recursive`
   - Passed
2. `bundle exec ruby -Itest test/onnx/extconf_compatibility_guard_test.rb`
   - `2 runs, 8 assertions, 0 failures, 0 errors`
3. `bundle exec ruby -Itest test/onnx/onnx_validation_test.rb`
   - `3 runs, 6 assertions, 0 failures, 0 errors`
4. `bundle exec ruby -Itest test/onnx/onnx_export_contract_boundary_test.rb`
   - `7 runs, 61 assertions, 0 failures, 0 errors`
5. `bundle exec ruby -Itest test/onnx/export_onnx_shapeless_facade_test.rb`
   - `4 runs, 30 assertions, 0 failures, 0 errors`
6. `bundle exec ruby -Itest test/onnx/onnx_native_timing_test.rb`
   - `3 runs, 31 assertions, 0 failures, 0 errors`
7. `bundle exec ruby -Itest test/onnx/docs_readme_onnx_refactor_drift_test.rb`
   - `2 runs, 40 assertions, 0 failures, 0 errors`

## Acceptance Criteria

1. Submodule gitlinks and `.gitmodules` are consistent.
2. Public API docs match current method names.
3. Internal symbol cleanup does not alter public Ruby method surface.
4. extconf compatibility guard is in place and tested.
5. CI explicitly checks submodule coherence and ONNX-native smoke paths.

## Risks and Mitigations

1. Risk: Renaming internal symbols accidentally changes exported method names.
   - Mitigation: keep `rb_define_singleton_method` strings unchanged; assert singleton method surface in tests.
2. Risk: Compatibility guard over-constrains local workflows.
   - Mitigation: only enforce when pinned revision is discoverable; clear remediation message.
3. Risk: CI time increase.
   - Mitigation: use targeted ONNX smoke tests, not full duplicate suite.

## Implementation Checklist

- [x] Phase 1 Red/Green complete.
- [x] Phase 2 Red/Green complete.
- [x] Phase 3 Red/Green complete.
- [x] Phase 4 Red/Green complete.
- [x] Phase 5 Red/Green complete.
- [x] Targeted test gates executed and passing.
