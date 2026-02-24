# 2026_02_23 Test Directory Cleanup PRD

Status: In Progress (2026-02-23)

## Context

The repository test suite has grown quickly and now mixes unit/integration intent, generated artifacts, long-lived parity phase files, and broad helper logic in `test/support/test_helper.rb`. The current layout increases maintenance cost and makes fast/slow behavior harder to reason about.

Current pressure points:

1. `test/support/test_helper.rb` owns too many unrelated responsibilities (native build, slow gating, export helpers, tmp helpers).
2. Test ownership boundaries are weak (`test/onnx`, `test/tasks`, `test/web`, `test/parity` all mix concerns).
3. Generated report artifacts live next to source-controlled test assets.
4. Fast/slow semantics are partly registry-based and partly implicit.
5. There are no layout guardrails to prevent future drift.

## Goals

1. Make test helper responsibilities modular and easy to reason about.
2. Organize tests by suite intent (`unit`, `integration`, `parity`, `tasks`, `docs`, etc.) with predictable paths.
3. Separate generated artifacts from curated/golden snapshots.
4. Make fast/slow selection deterministic and explicit.
5. Add automated layout lint checks to keep structure clean.

## Non-goals

1. Changing ONNX export semantics or runtime behavior.
2. Rewriting test logic unrelated to layout/ownership boundaries.
3. Renaming historical parity phase IDs used for traceability.
4. Replacing the parity suite architecture in this PRD.

## Phased Plan (Red/Green)

### Phase 1: Extract `test_helper` Responsibilities Into `test/support`

Red:
1. Add a failing test asserting key `TestSupport` methods are defined in dedicated `test/support/*` files instead of `test/support/test_helper.rb`.
2. Capture baseline failure showing method source locations still point at `test/support/test_helper.rb`.

Green:
1. Add support modules:
   - `test/support/native_build.rb`
   - `test/support/slow_tests.rb`
   - `test/support/export_helpers.rb`
   - `test/support/tmp_helpers.rb`
2. Wire `test/support/test_helper.rb` to compose these modules and keep public `TestSupport` method names unchanged.
3. Keep behavior identical.

Exit criteria:
1. New support-location test passes.
2. Existing helper-dependent tests remain green.

### Phase 2: Normalize Suite Directory Boundaries

Red:
1. Add a failing layout test defining allowed top-level suite directories and disallowed placements for new tests.

Green:
1. Introduce suite structure:
   - `test/unit/...`
   - `test/integration/...`
   - `test/parity/...`
   - `test/tasks/...`
   - `test/docs/...`
2. Move ONNX/runtime/browser tests into `unit` vs `integration` paths by intent.
3. Update `require_relative` paths and any path-based test tooling.

Exit criteria:
1. Layout policy test passes.
2. Moved tests run from new paths with no coverage loss.

### Phase 3: De-noise Parity File Naming With Manifest Mapping

Red:
1. Add a failing parity manifest test requiring a source-of-truth mapping file between phase IDs and test methods.

Green:
1. Add `test/parity/manifest.yml` mapping `phase_id -> file -> test method`.
2. Consolidate parity files by domain (`core`, `nn`, `optimizers`, `distributed`, `perf`) while preserving phase IDs in method names/comments.
3. Keep backward traceability in generated reports.

Exit criteria:
1. Manifest contract test passes.
2. Parity reporting continues to resolve phase IDs.

### Phase 4: Separate Generated Reports From Versioned Snapshots

Red:
1. Add a failing test that flags generated report output under source-controlled report directories.

Green:
1. Move runtime-generated outputs to `test/tmp/reports/`.
2. Keep curated baselines in `test/snapshots/`.
3. Update report-producing scripts/tests to read/write the correct locations.
4. Update `.gitignore` for generated artifacts.

Exit criteria:
1. Generated outputs are excluded from git-tracked source paths.
2. Snapshot-dependent tests use explicit snapshot paths.

### Phase 5: Make Fast/Slow Policy Explicit by Path + Registry

Red:
1. Add failing tests for fast-mode filtering rules (registry + forced slow path policy).

Green:
1. Extend slow gating policy to include explicit path classes (for heavy web/browser/integration suites) plus `slow_tests.json` entries.
2. Preserve existing opt-in (`MLX_TEST_INCLUDE_SLOW=1`, `rake test:all`).
3. Ensure `rake test:fast` remains deterministic on CPU-only local runs.

Exit criteria:
1. Fast/slow policy tests pass.
2. `rake test:fast` excludes intended heavy suites without ad hoc skips.

### Phase 6: Add Test Layout Guardrails

Red:
1. Add failing lint tests for:
   - test file naming convention
   - allowed directory placement
   - generated artifact placement
   - prohibition of new unmanaged `phaseNN_*` file additions (manifest required)

Green:
1. Implement `test/policy/layout_lint_test.rb`.
2. Add/refresh `test/reports/test_layout_manifest.txt` generation checks.

Exit criteria:
1. Layout lint test is green and enforced in fast suite.

### Phase 7: Documentation + Final Validation

Red:
1. Add/update docs tests that assert test command docs match current structure.

Green:
1. Update README/docs for new test layout and commands.
2. Run targeted-to-broad test gates.
3. Mark PRD completed when all phase exits and acceptance criteria are met.

Exit criteria:
1. Docs and test commands align.
2. Required gates are green.

## Test Execution Gates

1. Phase-targeted tests added in each phase (red -> green).
2. Helper and slow-gate targeted checks:
   - `bundle exec ruby -Itest test/policy/helper_support_split_test.rb`
   - `bundle exec ruby -Itest test/gem/build_policy_test.rb`
   - `bundle exec ruby -Itest test/policy/slow_test_gate_test.rb`
3. ONNX/unit smoke in touched areas:
   - `bundle exec ruby -Itest test/onnx/onnx_validation_test.rb`
   - `bundle exec ruby -Itest test/onnx/onnx_export_contract_boundary_test.rb`
4. Task-level checks for touched workflows:
   - `bundle exec rake test:fast MLX_TEST_DEVICES=cpu`
5. Full suite gate before completion:
   - `bundle exec rake test:all MLX_TEST_DEVICES=cpu`

If a gate cannot be run locally, record exactly what was skipped and why.

Execution results (so far):

1. `bundle exec ruby -Itest test/policy/helper_support_split_test.rb`
   - Red baseline: failed with 4 source-location assertions (methods still in `test/support/test_helper.rb`).
   - Green: passed (`4 runs, 12 assertions`).
2. `bundle exec ruby -Itest test/gem/build_policy_test.rb`
   - Passed (`3 runs, 11 assertions`).
3. `bundle exec ruby -Itest test/policy/slow_test_gate_test.rb`
   - Passed (`4 runs, 8 assertions`).
4. `bundle exec ruby -Itest test/onnx/onnx_validation_test.rb`
   - Passed (`3 runs, 6 assertions`).
5. `bundle exec ruby -Itest test/onnx/onnx_export_contract_boundary_test.rb`
   - Passed (`7 runs, 61 assertions`).
6. `bundle exec ruby -Itest test/policy/layout_policy_test.rb`
   - Red baseline: failed because web integration tests were still under `test/web/*`.
   - Green: passed (`1 runs, 10 assertions`) after moving web integration tests to `test/integration/web/*`.
7. `bundle exec ruby -Itest test/integration/web/gpt2_demo_integration_test.rb`
   - Passed (`1 runs, 9 assertions`).
8. `bundle exec ruby -Itest test/integration/web/nanogpt_demo_integration_test.rb`
   - Passed (`1 runs, 9 assertions`).
9. `bundle exec ruby -Itest test/integration/web/stable_diffusion_demo_integration_test.rb`
   - Passed (`1 runs, 13 assertions`).
10. `bundle exec ruby -Itest test/policy/layout_policy_test.rb`
   - Red baseline (Phase 2 ONNX slice 1): failed because ONNX runtime/browser/harness tests were still under `test/onnx/*`.
   - Green: passed after moving ONNX integration tests into `test/integration/onnx/*`.
11. `bundle exec ruby -Itest test/integration/onnx/real_runtime_smoke_dependency_gate_test.rb`
   - Passed (`2 runs, 10 assertions`).
12. `bundle exec ruby -Itest test/integration/onnx/export_onnx_runtime_test.rb`
   - Passed (`2 runs, 6 assertions`).
13. `bundle exec ruby -Itest test/integration/onnx/benchmark_model_onnx_runtime_test.rb`
   - Initial run exposed move regression (`repo_root` resolved to `.../test` and looked for `test/test/fixtures/karpathy.txt`).
   - Green after path fix (`repo_root: File.expand_path("../../..", __dir__)`): passed (`5 runs, 12 assertions, 1 skip`).
14. `bundle exec ruby -Itest test/integration/onnx/model_fixture_webgpu_browser_test.rb`
   - Passed with dependency/slow skips (`5 runs, 0 assertions, 5 skips`).
15. `bundle exec ruby -Itest test/policy/layout_policy_test.rb`
   - Red baseline (Phase 2 ONNX slice 2): failed because remaining WebGPU/report ONNX tests were still under `test/onnx/*`.
   - Green: passed after moving `onnx_webgpu_*` and `webgpu_python_metrics_test.rb` to `test/integration/onnx/*`.
16. `bundle exec ruby -Itest test/integration/onnx/onnx_webgpu_compat_report_test.rb`
   - Passed (`2 runs, 8 assertions`).
17. `bundle exec ruby -Itest test/integration/onnx/webgpu_python_metrics_test.rb`
   - Initial run exposed moved-path regression (`require_relative "../../tasks/*"`).
   - Green after path fix (`require_relative "../../../tasks/*"`): passed (`2 runs, 8 assertions`).
18. `bundle exec ruby -Itest test/policy/layout_policy_test.rb`
   - Red baseline (Phase 2 ONNX unit slice): failed because remaining ONNX unit tests were still under `test/onnx/*`.
   - Green: passed after moving ONNX unit tests to `test/unit/onnx/*`.
19. `bundle exec ruby -Itest test/unit/onnx/onnx_validation_test.rb`
   - Passed (`3 runs, 6 assertions`).
20. `bundle exec ruby -Itest test/unit/onnx/onnx_export_contract_boundary_test.rb`
   - Passed (`7 runs, 61 assertions`).
21. `bundle exec ruby -Itest test/unit/onnx/argreduce_lowering_test.rb`
   - Passed (`2 runs, 6 assertions`).
22. `bundle exec ruby -Itest test/unit/onnx/docs_readme_onnx_refactor_drift_test.rb`
   - Initial run exposed moved-path regression (`REPO_ROOT` derived from `__dir__`, resolving to `test/`).
   - Green after path fix (`REPO_ROOT = RUBY_ROOT`): passed (`2 runs, 40 assertions`).
23. `bundle exec ruby -Itest test/policy/layout_policy_test.rb`
   - Final Phase 2 verification after ONNX unit move: passed (`3 runs, 48 assertions`).
24. `bundle exec ruby -Itest test/unit/onnx/onnx_validation_test.rb`
   - Passed (`3 runs, 6 assertions`) after ONNX unit migration.
25. `bundle exec ruby -Itest test/unit/onnx/onnx_export_contract_boundary_test.rb`
   - Passed (`7 runs, 61 assertions`) after ONNX unit migration.
26. `bundle exec ruby -Itest test/unit/onnx/docs_readme_onnx_refactor_drift_test.rb`
   - Passed (`2 runs, 40 assertions`) after docs path updates and drift-test root fix.
27. `bundle exec ruby -Itest test/policy/helper_support_split_test.rb`
   - Passed (`4 runs, 12 assertions`) after moving helper/test-support files under `test/support/*`.
28. `bundle exec ruby -Itest test/policy/slow_test_gate_test.rb`
   - Passed (`6 runs, 14 assertions`) after moving slow-gate tests under `test/tasks/*`.
29. `bundle exec ruby -Itest test/policy/layout_lint_test.rb`
   - Passed (`5 runs, 10 assertions`) with the new no-top-level-`test/*.rb` guard.
30. `bundle exec ruby -Itest test/gem/build_policy_test.rb`
   - Passed (`3 runs, 11 assertions`) after moving gem tests under `test/gem/*`.
31. `bundle exec ruby -Itest test/tasks/test_task_slow_mode_test.rb`
   - Passed (`3 runs, 5 assertions`) after helper path updates.
32. `bundle exec ruby -Itest test/parity/phase0_manifest_test.rb`
   - Passed (`2 runs, 0 assertions, 2 skips`) after parity helper path updates.
33. `bundle exec ruby -Itest test/unit/onnx/onnx_validation_test.rb`
   - Passed (`3 runs, 6 assertions`) after ONNX helper path updates.
34. `bundle exec ruby -Itest test/dsl/dsl_test.rb`
   - Passed (`26 runs, 137 assertions`) after DSL helper path updates.
35. Full-suite rerun (`bundle exec rake test:all MLX_TEST_DEVICES=cpu`) was intentionally deferred after targeted fixes per user direction to rerun only failing/impacted tests during this phase.

## Acceptance Criteria

1. Test helper logic is modularized under `test/support/*` with unchanged public helper API.
2. Test suite directory boundaries are explicit and linted.
3. Parity phase traceability is preserved via manifest, without uncontrolled filename sprawl.
4. Generated artifacts are separated from versioned snapshots.
5. Fast/slow behavior is deterministic and documented.
6. Targeted and broad test gates pass.

## Risks and Mitigations

1. Risk: File moves break `require_relative` paths and selective test commands.
   - Mitigation: move incrementally; run targeted tests per move; keep stable aliases where needed.
2. Risk: Fast/slow policy accidentally hides critical regressions.
   - Mitigation: keep `test:all` gate required; enforce explicit slow classification tests.
3. Risk: Parity report tooling breaks with manifest/domain regrouping.
   - Mitigation: introduce manifest contract tests before refactor; preserve phase ID mapping.
4. Risk: Artifact path changes create noisy diffs.
   - Mitigation: enforce `.gitignore` and layout lint checks early.

## Implementation Checklist

- [x] Phase 1 Red/Green complete.
- [x] Phase 2 Red/Green complete.
- [x] Phase 3 Red/Green complete.
- [x] Phase 4 Red/Green complete.
- [x] Phase 5 Red/Green complete.
- [x] Phase 6 Red/Green complete.
- [ ] Phase 7 Red/Green complete.
- [ ] Final full-suite gate run and green.
