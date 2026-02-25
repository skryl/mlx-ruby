# MLX-Ruby Gap Fixes PRD

- Date: 2026-02-25
- Owner: Codex + @skryl
- Status: Completed (amended 2026-02-25 for safetensors default-on follow-up)
- Scope Type: Upstream `mlx-ruby` compatibility and ergonomics

## 1) Problem Statement

`mlx-ruby-lm` integration surfaced API/interop gaps in `mlx-ruby` that force repetitive workarounds, reduce parity with Python MLX, and increase model-porting cost.

Most impactful categories:

1. Core API ergonomics (`array(dtype:)`, reduction `keepdims`)
2. Ruby numeric coercion with `MLX::Core::Array`
3. `MLX::NN::Module` traversal/update behavior under nested structures
4. ONNX lowering parity (`GreaterEqual`)

## 2) Goals

1. Remove high-friction workarounds required across model codebases.
2. Preserve backward compatibility where possible.
3. Add regression tests for every accepted fix.
4. Keep docs explicit about unsupported/optional features.

## 3) Non-Goals

1. Rewriting `mlx-ruby-lm` model architectures in this PRD.
2. Full MoE feature parity beyond scoped API fixes (for example complete `SwitchGLU` implementation in this pass unless separately approved).
3. Changing CI packaging defaults unless explicitly approved.

## 4) Scope Matrix (From Reported Issues)

## 4.1 In Scope (this PRD)

1. Fix `MLX::Core.array(values, dtype: ...)` keyword compatibility.
2. Add `keepdims` support to `mean` (and align reductions where practical).
3. Add Ruby `coerce` support for `MLX::Core::Array` so `Float * array`, `Float + array`, etc. work.
4. Confirm/patch `update_modules_impl` recursion for `Module -> Hash/Array` replacement path.
5. Add ONNX lowering coverage for `GreaterEqual` (Issue 14).
6. Strengthen docs on module child registration (`self.x = ...` vs `@x = ...`).
7. Enable `MLX_BUILD_SAFETENSORS=ON` in this repo's native build defaults and require safetensors roundtrip tests to pass.

## 4.2 Out of Scope (tracked, not implemented in this PRD unless requested)

1. Add `SwitchGLU` layer implementation.
2. Dropout constructor API expansion (`Dropout.new(p: ...)`) unless explicitly requested for compatibility.
3. Integer dtype support in `random_uniform` unless upstream MLX behavior and API contract are aligned for deterministic semantics.

## 5) Detailed Requirements

1. `array(dtype:)`
   - Accept both positional dtype and keyword dtype.
   - Reject conflicting dtype values with clear error.
   - Keep existing behavior for `array(values)` and `array(values, dtype_obj)`.

2. `mean(..., keepdims:)`
   - Support `MLX::Core.mean(array, axis=nil, keepdims=false)` in Ruby API.
   - Support `MLX::Core::Array#mean(axis=nil, keepdims=nil)`.
   - Maintain existing no-axis behavior.

3. Ruby coercion
   - Implement `MLX::Core::Array#coerce(other)` so numeric-left operators work:
     - `1.5 * arr`
     - `1.5 + arr`
     - `2 - arr`
     - `2 / arr`
   - Add tests for `Float` and `Integer` on left-hand side.

4. Module update recursion
   - Ensure `NN.update_modules_impl` recursively handles:
     - current `Module` + replacement `Hash`
     - current `Module` + replacement `Array`
   - Preserve existing behavior for map/map and array/array recursion.

5. ONNX `GreaterEqual`
   - Add lowering support and integration/contract tests.
   - Confirm exported graph compatibility report no longer flags missing op in covered paths.

6. Documentation
   - Add explicit guidance in NN docs/README: assigning child modules must go through `self.child = ...` to register in state traversal.
   - Document safetensors compile-time optionality and fallback path.

## 6) Phased Plan and Checklist

## Phase 0: Baseline + Repro

- [x] Add or identify failing regression tests for each in-scope item.
- [x] Confirm currently failing behavior before code changes.
- [x] Update this PRD with confirmed repro status.

## Phase 1: Core API Fixes

- [x] Implement `array(dtype:)` keyword support in Ruby/native boundary.
- [x] Implement `mean(..., keepdims:)` support in Ruby/native boundary.
- [x] Add targeted unit/parity tests for both.

## Phase 2: Ruby Coercion + NN Traversal

- [x] Implement `Array#coerce` for numeric LHS ops.
- [x] Add coercion tests for add/sub/mul/div.
- [x] Patch/confirm `update_modules_impl` recursion behavior.
- [x] Add regression tests for module/hash/array update recursion.

## Phase 3: ONNX + Docs

- [x] Add `GreaterEqual` lowering support and tests.
- [x] Add docs for `self.x =` child registration requirement.
- [x] Add docs for safetensors optional build feature and fallback.

## Phase 4: Validation + Completion

- [x] Run targeted tests for touched files/features.
- [x] Run broader suite covering core/nn/onnx touched areas.
- [x] Safetensors default-on follow-up: flip build flag and verify native safetensors roundtrip behavior in parity tests.
- [x] Update PRD status to `Completed` only when all checklist items are done.

## 6.1) Baseline Repro Notes (2026-02-25)

Observed before fixes:

1. `MLX::Core.array([1,2,3], dtype: MLX::Core.int32)` raised `ArgumentError`.
2. `MLX::Core.mean(x, axis, keepdims: true)` raised wrong-arity `ArgumentError`.
3. Numeric-left ops (`1.5 + array`) raised `TypeError` (missing `coerce`).
4. `update_modules` raised `ArgumentError: Received invalid type: Hash.` for
   `current_value=Module` + `new_value=Hash` recursion paths.

Regression tests added:

1. `test/core/core_api_gap_regression_test.rb`
2. `test/nn/module_update_modules_recursion_test.rb`

## 7) Test Strategy

Minimum per-change targeted tests:

1. Core API:
   - `test/parity/phase5_core_ops_test.rb` (or dedicated new parity file)
   - new unit tests for `array(dtype:)` keyword and `mean keepdims`
2. Coercion:
   - new `test/parity` or `test/core` coverage for numeric-left operations
3. NN traversal:
   - existing quantization/state traversal tests plus new recursion regression test
4. ONNX:
   - ONNX binding/integration tests covering `GreaterEqual` export and runtime path

Completion sweep:

1. Run all targeted tests for modified files.
2. Run a broad cross-cutting suite including core + nn + onnx touched domains before marking complete.

## 8) Risks and Mitigations

1. Native binding signature changes can break call compatibility.
   - Mitigation: keep positional args valid; add explicit keyword parsing tests.
2. `coerce` may affect operator dispatch in edge cases.
   - Mitigation: limit support to numeric and raise clear `TypeError` otherwise.
3. ONNX lowering changes can affect compatibility reports.
   - Mitigation: add both positive tests and unsupported-op boundary tests.

## 9) Acceptance Criteria

1. All in-scope checklist items are checked.
2. New/updated tests for each implemented fix are green.
3. No regressions in touched core/nn/onnx paths.
4. PRD status changed from `Draft` to `Completed` only when all above are true.

## 10) Open Decisions

1. Should `Dropout.new(p: 0.5)` keyword support be included now or deferred?
2. Should integer `random_uniform` support be emulated in Ruby or left as explicit unsupported behavior?
3. Should safetensors default build flags be changed in CI/release pipelines in this effort? Resolved for this repo on 2026-02-25: native build default switched to `MLX_BUILD_SAFETENSORS=ON`.
