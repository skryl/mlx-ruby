# Benchmark Performance Remediation Plan (CNN + RNN)

## Goal

Close the Ruby vs Python performance gap for benchmark `cnn` and `rnn` while preserving output parity and benchmark path equivalence.

Current benchmark gap (GPU, `WARMUP=50`, `ITERATIONS=1000`):

- `cnn`: Ruby `0.929 ms` vs Python `0.436 ms` (`2.13x` slower)
- `rnn`: Ruby `7.468 ms` vs Python `4.178 ms` (`1.79x` slower)

## Issue Map

1. End-to-end benchmark gap in CNN/RNN.
2. CNN gap is mostly execution-path cost, not graph-build cost.
3. Ruby pooling implementation is slower than upstream path.
4. Ruby RNN has high graph-build and execution overhead.
5. Ruby recurrent kernels use less efficient op patterns than upstream.
6. Ruby activations are eager-only while Python uses compiled activation helpers.

## Phased Plan

### Phase 0: Baseline and Targets (Issues 1, 2, 4)

- Lock benchmark protocol and environment for reproducibility.
- Capture baseline:
  - full benchmark matrix (`benchmark`)
  - per-op microbench (`conv`, `relu`, `pool`, `linear`, `rnn`)
  - split timings (`build_only`, `build_plus_eval`)
- Define pass thresholds:
  - GPU `cnn` `rb/py <= 1.30x`
  - GPU `rnn` `rb/py <= 1.30x`

Deliverable:

- Committed baseline report and target table in `rfp/`.

### Phase 1: Pooling Fast Path Port (Issue 3)

- Port upstream pooling optimizations to Ruby:
  - non-overlapping window fast path
  - pooled reduction over all window axes in one operation
- Keep parity behavior identical.

Deliverable:

- Refactored `lib/mlx/nn/layers/pooling.rb` + test coverage.

Exit criteria:

- `maxpool2d` microbench improves by at least `25%`.
- CNN benchmark improves from Phase 0 baseline.

### Phase 2: Recurrent Op-Path Refactor (Issue 5)

- Refactor `RNN`, `GRU`, `LSTM` to use more fused operations where possible (`addmm` paths).
- Reduce per-step op count and avoid avoidable `take`/extra intermediate materialization patterns.

Deliverable:

- Refactored `lib/mlx/nn/layers/recurrent.rb` + parity tests.

Exit criteria:

- `rnn_full` microbench improves by at least `20%`.
- No parity regressions in benchmark checks.

### Phase 3: Ruby Graph-Build Overhead Reduction (Issue 4)

- Reduce Ruby-side loop and object churn in hot model paths.
- Cache static metadata and eliminate repeated Ruby work in inner loops where safe.

Deliverable:

- Profiling-backed changes with before/after timing report.

Exit criteria:

- `rnn_build_only` improves by at least `30%`.
- `cnn_build_only` improves where measurable.

### Phase 4: Activation Compile Parity (Issue 6)

- Add compiled activation path in Ruby analogous to upstream activation usage.
- Preserve eager fallback behavior.

Deliverable:

- Activation runtime improvements + regression tests.

Exit criteria:

- Activation microbench near Python parity.
- Measurable CNN improvement versus Phase 0.

### Phase 5: End-to-End Validation and Guardrails (Issue 1 Closure)

- Re-run full benchmark matrix with Phase 0 protocol.
- Publish before/after summary.
- Update `README.md` benchmark table.
- Add non-flaky perf guardrails in CI (warning-first, then fail on sustained regression).

Deliverable:

- Final benchmark report + README update + CI threshold checks.

Exit criteria:

- CNN and RNN meet target ratios or have documented residual blockers with quantified delta.

## Execution Strategy

- Ship by phase in small PRs to isolate regressions.
- Require parity check pass (`input_shape`, `input_digest`, `output_shape`, `reference_output_digest`) for each phase.
- Treat benchmark speedup claims as valid only when measured with fixed protocol from Phase 0.

## Progress Update (2026-02-17)

Completed so far:

- Phase 1 (partial): pooling non-overlapping sliding window fast path and pooled-axis reduction for max.
- Phase 2 (partial): recurrent refactor (`addmm` paths, transpose caching, while-loop/preallocated hidden buffers, direct state fetches).
- Phase 3 (partial): reduced Ruby dispatch overhead in hot paths (`Conv*`, `Linear`, `Bilinear`, benchmark runner proc fast path).
- Additional RNN optimization: compiled hidden-state update fast path for default tanh recurrence with safe eager fallback.

Validation:

- Parity tests passing for updated areas:
  - `test/parity/phase187_linear_layers_parity_test.rb`
  - `test/parity/phase193_convolution_layers_parity_test.rb`
  - `test/parity/phase195_pooling_layers_parity_test.rb`
  - `test/parity/phase197_recurrent_layers_parity_test.rb`
  - `test/parity/phase190_activations_parity_test.rb`

Latest GPU benchmark samples (`WARMUP=50`, `ITERATIONS=1000`):

- `rnn`: Ruby `3.687 ms`, Python `4.219 ms` (Ruby faster in current run)
- `cnn`: Ruby `0.647 ms`, Python `0.397 ms` (Ruby slower; residual gap remains)

Current blocker:

- CNN still trails despite op-path and dispatch optimizations. Residual gap appears concentrated in `conv + relu + pool` execution cost under Ruby wrapper calls; needs deeper profiling/fusion strategy in Phase 3/4 follow-up.
