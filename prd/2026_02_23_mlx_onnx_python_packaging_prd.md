# 2026_02_23 MLX ONNX Python Packaging PRD

Status: Completed (2026-02-23)

## Context

`mlx-onnx` was extracted as a standalone native IR/ONNX repo, but it still needed a proper pip-installable Python package surface so Python consumers can install and use the public API directly.

## Goals

1. Make `mlx-onnx` installable with `pip install <wheel>` and `pip install -e`.
2. Expose the IR/ONNX public API from a package namespace (`mlx_onnx`).
3. Ensure build reliability on macOS in this environment.
4. Keep wheel contents focused on runtime Python artifacts.
5. Validate package behavior with automated tests and runtime smoke checks.

## Non-goals

1. Publishing to PyPI in this phase.
2. Reworking IR lowering semantics.
3. Adding browser/web harness tests.

## Phased Plan (Red/Green)

### Phase 1: Packaging skeleton

Red:
1. No `pyproject.toml` packaging metadata for `mlx-onnx`.
2. No installable `mlx_onnx` package namespace.

Green:
1. Add `mlx-onnx/pyproject.toml` with `scikit-build-core`.
2. Add `python/mlx_onnx/__init__.py` public re-exports.
3. Build Python extension module as `mlx_onnx._core`.

Exit criteria:
1. `python -m build --wheel mlx-onnx` can produce a wheel artifact.

### Phase 2: Build reliability and install shape

Red:
1. Wheel link failure from Homebrew LLVM toolchain mismatch (`__hash_memory` unresolved).
2. Wheel bundles non-runtime MLX dev artifacts.

Green:
1. Force Apple clang in packaging CMake args.
2. Add CMake install components and configure wheel install to `python` component only.
3. Keep optional C++ artifact installs available behind `MLX_ONNX_INSTALL_CPP_ARTIFACTS`.

Exit criteria:
1. Wheel build succeeds in this environment.
2. Wheel contains only package runtime files (`mlx_onnx/*`, dist-info).

### Phase 3: Test alignment and validation

Red:
1. `python/tests/test_ir.py` depended on `mlx_tests` and `mx.export_*` in-core API.

Green:
1. Update tests to import and exercise `mlx_onnx` package API.
2. Add fallback base test case when `mlx_tests` is unavailable.
3. Re-run test suite against installed package.

Exit criteria:
1. `python -m unittest mlx-onnx/python/tests/test_ir.py` passes (except optional skips).

## Test Execution Gates

1. `tmp/venv_mlx_onnx/bin/python -m build --wheel mlx-onnx`
   - Result: success, wheel created.
2. `tmp/venv_mlx_onnx_pkg2/bin/pip install mlx-onnx/dist/mlx_onnx-0.1.0-cp314-cp314-macosx_26_0_arm64.whl`
   - Result: success, dependency resolution includes `mlx`.
3. Runtime smoke:
   - `import mlx_onnx`
   - `export_ir_json`, `export_onnx_json`, `export_onnx` on a simple MLX function
   - Result: success.
4. `tmp/venv_mlx_onnx_pkg2/bin/pip install -e mlx-onnx`
   - Result: success.
5. `tmp/venv_mlx_onnx_pkg2/bin/python -m unittest mlx-onnx/python/tests/test_ir.py`
   - Result: `Ran 24 tests ... OK (skipped=4)`.

## Acceptance Criteria

1. `mlx-onnx` is pip-installable as wheel and editable install.
2. Public package API is exposed under `mlx_onnx`.
3. Packaging build is stable in the current macOS toolchain setup.
4. Python test suite for touched areas passes.

## Risks and Mitigations

1. Risk: toolchain mismatch on macOS causes native link failures.
   - Mitigation: pin CMake compilers to `/usr/bin/clang` and `/usr/bin/clang++` for package builds.
2. Risk: oversized wheels from transitive install rules.
   - Mitigation: use CMake install components and scikit-build install component filtering.
3. Risk: test suite coupling to in-core MLX test harness.
   - Mitigation: fallback to `unittest.TestCase` and route tests through `mlx_onnx` API.

## Implementation Checklist

- [x] Phase 1 Red/Green complete.
- [x] Phase 2 Red/Green complete.
- [x] Phase 3 Red/Green complete.
- [x] Wheel build/install smoke complete.
- [x] Editable install smoke complete.
- [x] Targeted Python tests green.
