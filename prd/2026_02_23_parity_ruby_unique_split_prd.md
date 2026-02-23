# 2026_02_23_parity_ruby_unique_split_prd

## Status
In Progress

Blocked on upstream test failures unrelated to file moves (see current validation results).

## Context
`test/parity/` currently mixes tests that validate shared MLX behavior and tests that validate Ruby-only surfaces (notably GraphIR/ONNX/WebGPU pipeline and DSL-adapter behavior). The target policy is:

1. Keep `test/parity/` focused on behavior that exists in both official Python MLX and mlx-ruby.
2. Move Ruby-unique parity coverage to module-owned test directories.

## Goals
1. Move Ruby-unique GraphIR/ONNX/WebGPU parity files from `test/parity/` to `test/graph_ir/`.
2. Move Ruby-unique DSL adapter parity file to `test/dsl/`.
3. Preserve test behavior, class names, and phase filenames.
4. Produce a full post-move test inventory under the new structure.

## Non-goals
1. Rewriting test assertions/semantics.
2. Deleting tests.
3. Renaming phase classes/files.
4. Refactoring parity scripts/reports beyond path updates required by this move.

## Phased Plan

### Phase 1: Classify and lock move set
#### Red
1. Capture current `test/parity/phase*_test.rb` inventory.
2. Verify which files are Ruby-unique via content signals (`MLX::GraphIR`, ONNX/WebGPU harness APIs, DSL-mode adapter behavior).

#### Green
1. Freeze the explicit move set (63 files).
2. Define destination layout:
   - `test/graph_ir/` for GraphIR/ONNX/WebGPU tests.
   - `test/dsl/` for DSL adapter phase test.

#### Exit criteria
1. Move list and destinations are explicit and deterministic.
2. No unresolved file classification decisions remain.

### Phase 2: Execute file moves and helper wiring
#### Red
1. Add/prepare a loader path so moved GraphIR tests no longer depend on `test/parity/test_helper.rb` being adjacent.

#### Green
1. Create `test/graph_ir/test_helper.rb` that requires `../test_helper`.
2. Move 62 GraphIR files from `test/parity/` to `test/graph_ir/`.
3. Move `examples_adapter_boolean_diff_test.rb` to `test/dsl/` and update it to require `../test_helper`.
4. Keep remaining `test/parity/` files unchanged.

#### Exit criteria
1. All listed files exist only at their new paths.
2. Moved files load test helper successfully.

### Phase 3: Reference updates, validation, and inventory report
#### Red
1. Search for stale hardcoded references to old `test/parity/<moved-file>` paths.

#### Green
1. Update any stale references if found.
2. Run targeted test gates:
   - `bundle exec ruby -Itest test/dsl/examples_adapter_boolean_diff_test.rb`
   - `TEST='test/graph_ir/**/*_test.rb' bundle exec rake "test[cpu]"`
3. Run broader suite gate:
   - `bundle exec rake test`
4. Generate full post-move inventory artifact of `test/**/*_test.rb`.

#### Exit criteria
1. Required tests pass (or skips/failures are documented with reasons).
2. Inventory artifact is generated and complete.
3. PRD checklist reflects actual completion state.

## Acceptance Criteria
1. Exactly the approved 63 Ruby-unique files are moved from `test/parity/`.
2. `test/parity/` no longer contains GraphIR/ONNX/WebGPU pipeline tests or DSL adapter parity test.
3. Remaining `test/parity/` continues to cover shared Python/Ruby functionality.
4. Targeted and broad test gates are executed and documented.
5. Full post-move test inventory is produced.

## Risks and mitigations
1. Risk: Broken helper resolution after move.
   Mitigation: Add `test/graph_ir/test_helper.rb` shim and run targeted loader tests.
2. Risk: Hidden references to old paths in tasks/docs/tests.
   Mitigation: repo-wide grep for moved filenames and `test/parity/` path patterns.
3. Risk: Long-running parity graph tests depend on optional Python/runtime tools.
   Mitigation: run required gates; document environment-based skips if present.

## Implementation checklist
- [x] Phase 1 Red: Capture parity inventory and classify Ruby-unique files.
- [x] Phase 1 Green: Freeze move set and destination structure.
- [x] Phase 2 Red: Prepare helper-loading approach for moved GraphIR tests.
- [x] Phase 2 Green: Move files and apply helper/path updates.
- [x] Phase 3 Red: Search for stale old-path references.
- [ ] Phase 3 Green: Run targeted and broad test gates.
- [x] Phase 3 Green: Generate full post-move inventory artifact.
- [ ] Final: Mark PRD as `Completed (2026-02-23)` only after all phases are done.

## Current validation results
1. `bundle exec ruby -Itest test/dsl/examples_adapter_boolean_diff_test.rb`
   - Passed.
2. `TEST='test/graph_ir/**/*_test.rb' bundle exec rake "test[cpu]"`
   - Failed in `test/graph_ir/complex64_initializer_lowering_test.rb` with 2 failures and 2 errors.
3. `bundle exec rake test`
   - Aborted by upstream native crash (`SIGSEGV`) in `test/parity/phase250_load_save_edge_parity_test.rb`.
