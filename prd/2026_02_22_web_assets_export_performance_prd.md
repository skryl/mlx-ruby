# 2026_02_22 Web Assets Export Performance PRD

## Status

Completed (2026-02-22)

## Context

`bundle exec rake web:assets` is currently slow and, in large ONNX cases (GPT-2),
can fail during ONNX materialization due to large JSON transport through
`Open3.capture3(stdin_data: ...)`.

Baseline signal observed locally:
1. GPT-2 exporter step consumed ~292s wall time.
2. Failure occurred in `MLX::GraphIR::ONNX::PythonBuilder.run_python!` while
   writing large stub JSON over stdin (`Errno::EINVAL`).

The user requirement is:
1. Asset generation must still regenerate on every run.
2. The generation itself should be faster and benchmarked per model.

## Goals

1. Remove fragile/slow stdin transport for large ONNX stubs.
2. Improve export-stage throughput where safe without changing artifact contracts.
3. Emit explicit per-model timing benchmarks in the `web:assets` flow.
4. Run `web:assets` to completion and capture benchmark numbers for all models.

## Non-goals

1. Skipping regeneration based on cache/up-to-date checks.
2. Changing default demo model repos or replacing model architectures.
3. Altering required output filenames/contracts consumed by web demos.

## Phased Plan (Red/Green)

### Phase 1: ONNX transport failure guard

Red:
1. Add a failing contract test proving `onnx_json_to_onnx` should use file-based
   stub transport (not stdin transport) for builder invocation.

Green:
1. Update ONNX exporter to write stub JSON to a temp file and invoke file-path
   Python builder path.
2. Keep output path/external data behavior unchanged.

Exit criteria:
1. New contract test passes.
2. Existing ONNX export parity tests remain green.

### Phase 2: Exporter runtime improvements

Red:
1. Capture benchmark timings from current exporters to identify dominant stages.

Green:
1. Add per-stage timing instrumentation to exporter scripts.
2. Apply safe runtime knobs that reduce export overhead while preserving outputs.

Exit criteria:
1. Export logs include clear stage timings per model.
2. At least one dominant stage shows reduced wall time.

### Phase 3: End-to-end validation + benchmark report

Red:
1. `web:assets` baseline fails or exceeds acceptable turnaround.

Green:
1. Run `bundle exec rake web:assets` end-to-end after changes.
2. Provide benchmark summary for GPT-2, Stable Diffusion, and nanoGPT.

Exit criteria:
1. `web:assets` completes successfully.
2. Benchmarks for all models are captured in the run output.

## Acceptance Criteria

1. Large-model ONNX conversion path no longer relies on stdin JSON transport.
2. `web:assets` prints per-model benchmark timings.
3. `web:assets` runs to completion with regenerated assets.
4. Output asset contract remains compatible with current web demos/tests.

## Risks and Mitigations

1. Risk: file-based stub transport may alter builder assumptions.
   Mitigation: keep existing Python builder entrypoint and add contract tests.
2. Risk: performance knobs could change model runtime compatibility.
   Mitigation: keep ONNX opset/output names/contracts unchanged; validate task and
   targeted tests.
3. Risk: long Stable Diffusion export may still dominate.
   Mitigation: benchmark each phase and isolate dominant sub-steps for future
   optimizations.

## Implementation Checklist

- [x] Phase 1 red: add transport contract test.
- [x] Phase 1 green: switch ONNX stub transport to temp file path.
- [x] Phase 1 exit criteria met.
- [x] Phase 2 red: baseline timings captured.
- [x] Phase 2 green: apply runtime optimizations + stage benchmarks.
- [x] Phase 2 exit criteria met.
- [x] Phase 3 red: baseline failure/slow signal documented.
- [x] Phase 3 green: full `web:assets` run completes with benchmarks.
- [x] Phase 3 exit criteria met.
