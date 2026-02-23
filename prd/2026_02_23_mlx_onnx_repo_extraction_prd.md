# 2026_02_23 MLX ONNX Repo Extraction PRD

Status: Completed (2026-02-23)

## Context

IR/ONNX native code was living in the `mlx` submodule working tree (`mlx/mlx/ir*`) while Ruby bindings consumed it from `mlx-ruby`.
The goal is to make `mlx-onnx` the standalone source of truth, with this repo consuming it as a dependency.

## Goals

1. Stand up `mlx-onnx` as a real buildable repository with its own `mlx` submodule.
2. Move IR/ONNX C++ core implementation ownership to `mlx-onnx`.
3. Move Python IR binding/test ownership artifacts into `mlx-onnx`.
4. Rewire Ruby native build to link against `mlx-onnx` (`mlx_onnx_ir`) instead of in-tree `mlx` IR sources.
5. Remove legacy in-tree IR ownership from `mlx` in this workspace.

## Non-goals

1. Full Python packaging parity for `mlx.core` inside `mlx-onnx` in this change.
2. Re-designing Ruby public IR APIs.
3. Semantic changes to lowering/compatibility/binary generation.

## Phased Plan (Red/Green)

### Phase 1: Create standalone `mlx-onnx` build skeleton

Red:
1. `mlx-onnx` contains only README and cannot build/link IR core.

Green:
1. Add `mlx-onnx/CMakeLists.txt`.
2. Add nested `mlx` submodule under `mlx-onnx/mlx`.
3. Add installable shared target `mlx_onnx_ir`.
4. Support both:
   - standalone mode (`mlx-onnx/mlx`)
   - external-MLX mode (include/lib dir inputs)

Exit criteria:
1. `mlx-onnx` has a concrete CMake build graph for IR core.

### Phase 2: Move IR core code ownership to `mlx-onnx`

Red:
1. IR core files only exist in `mlx/mlx/ir*`.

Green:
1. Copy IR core to `mlx-onnx/include/mlx/ir.hpp` and `mlx-onnx/src/ir/*`.
2. Fix header boundaries for standalone include layout.

Exit criteria:
1. IR core sources are present in `mlx-onnx` and build target sources reference them.

### Phase 3: Move Python IR ownership artifacts

Red:
1. Python IR binding/test files only exist in `mlx/python/...`.

Green:
1. Copy `mlx/python/src/ir.cpp` to `mlx-onnx/python/src/ir.cpp`.
2. Copy `mlx/python/tests/test_ir.py` to `mlx-onnx/python/tests/test_ir.py`.
3. Add lightweight Python module CMake entrypoint (`mlx-onnx/python/src/module.cpp`, `mlx-onnx/python/src/CMakeLists.txt`).

Exit criteria:
1. Python IR source/test ownership exists in `mlx-onnx`.

### Phase 4: Rewire Ruby native build to consume `mlx-onnx`

Red:
1. `ext/mlx/extconf.rb` only builds/links `mlx` and expects IR code in that tree.

Green:
1. Update `ext/mlx/extconf.rb` to:
   - build/install `mlx` as before,
   - configure/build/install `mlx-onnx` in external-MLX mode,
   - include `mlx-onnx` include/src roots,
   - link `-lmlx_onnx_ir -lmlx`.

Exit criteria:
1. Ruby extension builds through tests while resolving IR symbols from `mlx_onnx_ir`.

### Phase 5: Remove legacy in-tree `mlx` IR ownership

Red:
1. Local `mlx` submodule still carries in-tree `ir` additions and Python IR additions.

Green:
1. Restore `mlx` CMake/python binding files to baseline.
2. Remove local-added IR/Python IR files from `mlx` working tree.
3. Update IR boundary test paths to `mlx-onnx` locations.

Exit criteria:
1. `mlx` submodule is clean (no local IR ownership changes).
2. Boundary tests assert files under `mlx-onnx`.

## Test Execution Gates

1. `bundle exec ruby -Itest test/ir/graph_ir_core_boundary_style_test.rb`
   - Result: `5 runs, 26 assertions, 0 failures`
2. `bundle exec ruby -Itest test/ir/graph_ir_validation_test.rb`
   - Result: `3 runs, 6 assertions, 0 failures`
3. `bundle exec ruby -Itest test/ir/export_onnx_binary_test.rb`
   - Result: `3 runs, 13 assertions, 0 failures`
4. `bundle exec ruby -Itest test/ir/export_onnx_external_data_test.rb`
   - Result: `4 runs, 17 assertions, 0 failures`

## Acceptance Criteria

1. `mlx-onnx` has its own `mlx` submodule and CMake build.
2. IR core code is no longer owned by local `mlx` submodule changes.
3. Ruby native build links against `mlx_onnx_ir`.
4. IR Ruby tests in touched areas pass.

## Risks and Mitigations

1. Risk: Header include drift after move.
   - Mitigation: keep `mlx::core::ir` namespace + `mlx/ir.hpp` public path, adjust internal includes.
2. Risk: Duplicate/competing IR implementations.
   - Mitigation: clean `mlx` local IR additions; route extension to `mlx-onnx` only.
3. Risk: Build graph fragility with nested submodules.
   - Mitigation: explicit external-MLX mode wiring in `extconf.rb`.

## Implementation Checklist

- [x] Phase 1 Red/Green complete.
- [x] Phase 2 Red/Green complete.
- [x] Phase 3 Red/Green complete.
- [x] Phase 4 Red/Green complete.
- [x] Phase 5 Red/Green complete.
- [x] Targeted IR gates executed and passing.
