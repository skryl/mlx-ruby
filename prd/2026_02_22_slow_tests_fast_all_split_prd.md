# Slow Tests Fast/All Split PRD

Status: Completed (2026-02-21)

## Context

The current test entrypoint (`rake test`) runs the entire suite on both cpu and gpu and includes long-running scenarios. This slows local iteration and does not provide an explicit fast-vs-all contract.

We need:

1. Method-level timing visibility for the suite.
2. Deterministic classification of slow tests (`> 30s`).
3. Fast local default (`rake test` / `rake test:fast`) that excludes slow tests.
4. Full-suite mode (`rake test:all`) that includes slow tests.
5. CI to always run full coverage (`rake test:all`).

## Goals

1. Time every test method (cpu and gpu verbose runs).
2. Generate checked-in slow-test registry.
3. Enforce slow-test skipping in fast mode only.
4. Add clear rake tasks for fast vs all.
5. Update CI workflows to run all tests including slow.

## Non-Goals

1. Reorganizing parity vs non-parity test directories.
2. Changing existing test logic unrelated to slow-gating.
3. Introducing flaky-test retry behavior.

## Phased Plan

### Phase 1: Timing profiler and slow registry (Red/Green)

Red:

1. Add parser/generator tests for verbose timing logs and slow registry output.

Green:

1. Add timing profiler script and reusable parser helpers.
2. Generate `test/slow_tests.json` from cpu+gpu timing logs using max(cpu,gpu).

Exit criteria:

1. Timing parser tests pass.
2. Slow registry format is deterministic and checked in.

### Phase 2: Slow-test gating (Red/Green)

Red:

1. Add tests proving marked slow tests are skipped in fast mode and included in all mode.

Green:

1. Add slow-test registry loader in `test/test_helper.rb`.
2. Apply skip in `Minitest::Test#before_setup` when `MLX_TEST_INCLUDE_SLOW != "1"`.

Exit criteria:

1. Gating tests pass.
2. Fast runs skip marked slow tests with actionable message.

### Phase 3: Task surface split (Red/Green)

Red:

1. Add task-smoke expectations for `test:fast`, `test:all`, `test:cpu_all`, `test:gpu_all`.
2. Add task-level behavior tests for include-slow env toggling.

Green:

1. Update `MlxTestTask` to run with include-slow toggle.
2. Update `Rakefile` tasks:
   - `test` => fast default
   - `test:fast` => fast explicit
   - `test:all` => include slow
   - `test:cpu_all`/`test:gpu_all` => include slow for single device

Exit criteria:

1. New tasks are wired and passing smoke tests.
2. `rake test` and `rake test:fast` semantics match.

### Phase 4: CI wiring (Green)

1. Update CI workflows to call `bundle exec rake test:all`.

Exit criteria:

1. CI configs point at full-suite task.

## Acceptance Criteria

1. A checked-in `test/slow_tests.json` exists and uses threshold `30s`.
2. `rake test` skips slow tests.
3. `rake test:fast` behaves the same as `rake test`.
4. `rake test:all` includes slow tests.
5. CI workflows run `rake test:all`.
6. Targeted tests for parser, gating, and task surface are green.

## Risks and Mitigations

1. Risk: runtime-dependent slow list drift across environments.
   - Mitigation: deterministic generation using max(cpu,gpu), checked-in baseline.
2. Risk: false-positive skips for renamed tests.
   - Mitigation: keyed by method id; missing ids are ignored.
3. Risk: regression in existing task behavior.
   - Mitigation: expand task smoke tests and add task behavior unit tests.

## Implementation Checklist

- [x] Phase 1 Red: add timing parser/registry tests
- [x] Phase 1 Green: implement profiler and generate registry
- [x] Phase 2 Red: add slow-test gating tests
- [x] Phase 2 Green: implement gating in test helper
- [x] Phase 3 Red: add task surface/behavior tests
- [x] Phase 3 Green: implement task split and include-slow toggle
- [x] Phase 4 Green: update CI workflows to run full suite
- [x] Run targeted test gates and report outcomes
- [x] Run full suite regression gate
- [x] Mark PRD Completed with date
