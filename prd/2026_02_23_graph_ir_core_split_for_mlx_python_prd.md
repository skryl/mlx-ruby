# 2026_02_23 GraphIR Core Split for MLX Python PRD

## Status
In Progress (Phases 1-3 completed in-repo on 2026-02-23; awaiting upstream merge workflow)

## Context
`ext/mlx/graph_ir_native.cpp` currently mixes GraphIR capture, ONNX lowering, compatibility probing, ONNX protobuf encoding, file writing, and Ruby binding glue in one translation unit.

To upstream GraphIR support into `mlx` (and expose it in Python), we need a reusable core layer that is not coupled to Ruby types.

## Goals
1. Split native GraphIR code into core vs Ruby-specific binding layers in `mlx-ruby`.
2. Refactor the extracted core layer to follow `mlx` core-style structure and naming conventions.
3. Add explicit intermediate pre-upstream phases so the extracted core matches `mlx` conventions before repo migration.
4. Gate upstream merge and Python bindings on completion of all intermediate style/refactor phases.

## Non-goals
1. Final upstream merge/release process outside this repo.
2. Changing the Ruby public API shape.
3. Altering ONNX lowering semantics or supported-op coverage.

## Phased Plan (Red/Green)

### Phase 1: Core split in mlx-ruby
Red:
1. Add/adjust tests that enforce Ruby API behavior remains unchanged when core logic moves out of Ruby binding file.
2. Capture failing signal if split introduces API or serialization drift.

Green:
1. Introduce `ext/mlx/graph_ir_core.hpp/.cpp` with reusable C++ core for:
   - GraphIR -> ONNX JSON lowering
   - compatibility report generation
   - ONNX binary artifact building
   - binary artifact file writing
2. Keep Ruby conversion/argument parsing/binding registration in `graph_ir_native.cpp`.
3. Rewire Ruby entrypoints to call the new core layer.

Exit criteria:
1. Ruby GraphIR native tests pass.
2. Ruby public methods still return the same types and payload structure.

### Phase 2: Refactor core in mlx-ruby to mlx-style
Red:
1. Capture any style/structure mismatches vs expected `mlx` core patterns in current extraction.

Green:
1. Refactor extracted core module organization/naming to mirror `mlx` core conventions:
   - explicit namespace-scoped core API
   - helper grouping for lowering/encoding/io
   - typed options/artifact structs in header boundary
2. Minimize Ruby-era assumptions at core boundary.
3. Keep behavior unchanged.

Exit criteria:
1. Core code is transplant-ready for `mlx` with minimal mechanical move.
2. Tests remain green after refactor.

### Phase 2.5: Pre-Phase-3 style delta baseline
Red:
1. Capture concrete deltas between current `graph_ir_core` and upstream `mlx` core conventions.
2. Identify public API, namespace, file-layout, and IO/error handling mismatches that would add migration friction.

Green:
1. Produce an actionable sequence for namespace/API/layout/error/style/test alignment.
2. Define phase-level test gates for each intermediate refactor.

Exit criteria:
1. A migration-ready style delta checklist exists and is validated against upstream `mlx` patterns.
2. Each intermediate refactor is represented as an explicit phase prior to upstream merge.

### Phase 2.6: Subsystem layout and namespace alignment (pre-upstream)
Red:
1. Add checks that fail if GraphIR core symbols/files remain outside the intended subsystem namespace/layout plan.

Green:
1. Align namespace target with upstream convention (`mlx::core::graph_ir`).
2. Align file organization with subsystem-style layout target for upstream move (`mlx/mlx/graph_ir/...` map).
3. Remove transitional namespace aliases once call sites are updated.

Exit criteria:
1. Core namespace/layout plan matches upstream subsystem conventions.
2. Existing behavior and tests remain unchanged.

### Phase 2.7: Public API boundary typing and JSON-edge isolation
Red:
1. Add checks ensuring public headers do not expose third-party JSON types.

Green:
1. Keep JSON parsing/serialization at API edges only.
2. Define typed boundary structs for core entrypoints at the stable API surface.
3. Ensure Ruby/Python bindings interact with typed boundaries, not raw JSON internals.

Exit criteria:
1. Public headers avoid direct JSON dependency leakage.
2. API boundary is typed and binding-friendly for upstream reuse.

### Phase 2.8: Core decomposition into focused translation units
Red:
1. Add/adjust tests to detect regressions while splitting files.

Green:
1. Split monolithic core implementation into focused units:
   - types/shared utilities
   - lowering
   - compatibility reporting
   - ONNX protobuf encoding
   - artifact IO
2. Keep helper internals private and phase-local.

Exit criteria:
1. File decomposition is complete and behavior remains equivalent.
2. Build and tests remain green.

### Phase 2.9: Public/internal header layering and detail hygiene
Red:
1. Add checks to prevent internal detail helpers from leaking into public headers.

Green:
1. Introduce clear public-vs-internal header layering (`api` boundary vs `detail` internals).
2. Keep non-stable helpers in internal headers/translation units only.
3. Match upstream header ergonomics (`MLX_API`-oriented public entrypoints).

Exit criteria:
1. Header layering is explicit and maintainable.
2. Public API is stable and minimal.

### Phase 2.10: Error/diagnostic convention normalization
Red:
1. Add assertions around expected error class/tag shapes where externally relied upon.

Green:
1. Normalize error tags/messages to consistent mlx-style prefixes (e.g. `[graph_ir]`, `[graph_ir.lowering]`).
2. Keep unsupported-op signaling distinguishable from generic validation/runtime failures.

Exit criteria:
1. Errors are consistently tagged and actionable.
2. Unsupported coverage reporting remains stable.

### Phase 2.11: Style/format parity with upstream mlx core
Red:
1. Capture formatting/style drift against upstream `.clang-format`.

Green:
1. Apply upstream formatting and include ordering to GraphIR core files.
2. Remove residual Ruby-era section/comment patterns from core implementation where not needed.
3. Keep comments sparse and intent-focused in mlx style.

Exit criteria:
1. Core files conform to upstream formatting/style conventions.
2. No behavioral changes from style-only edits.

### Phase 2.12: Upstream-facing parity test plan finalization
Red:
1. Identify missing upstream-side parity coverage before migration.

Green:
1. Define upstream Python parity matrix tied to the six native interface methods.
2. Reuse/port ruby-side oracle comparisons where applicable (including ONNX binary parity).

Exit criteria:
1. Upstream parity plan is complete and implementation-ready for Phase 3.
2. Migration risks are explicitly tracked with tests.

### Phase 2.13: Lowering modular split hardening
Red:
1. Add a boundary test that fails until focused lowering/ONNX module files exist.

Green:
1. Split lowering into focused translation units:
   - `graph_ir_core_lowering_args.cpp`
   - `graph_ir_core_lowering_shape.cpp`
   - `graph_ir_core_lowering_ops.cpp`
   - `graph_ir_core_lowering_dispatch.cpp`
2. Keep graph-level wrapper/public payload path in dedicated dispatch unit.

Exit criteria:
1. Lowering no longer depends on a monolithic translation unit.
2. Existing lowering behavior remains stable under targeted tests.

### Phase 2.14: ONNX encoder decomposition (wire/tensor/assembly)
Red:
1. Use boundary checks to enforce file-level encoder decomposition.

Green:
1. Split ONNX binary encoding into focused units:
   - `graph_ir_core_onnx_wire.cpp`
   - `graph_ir_core_onnx_tensor.cpp`
   - `graph_ir_core_onnx_assemble.cpp`
2. Keep `graph_ir_core_onnx_encode.cpp` as a lightweight aggregation unit.

Exit criteria:
1. Encoder decomposition is complete and extension links cleanly.
2. ONNX binary output tests remain green.

### Phase 2.15: Centralized mapping/lookup tables
Red:
1. Remove duplicated mapping logic from lowering/encoder units.

Green:
1. Add shared mapping module:
   - `graph_ir_core_mappings.hpp`
   - `graph_ir_core_mappings.cpp`
2. Route GraphIR->ONNX op mappings, reduction mappings, dtype promotion rank lookups, and ONNX element symbol lookup through shared mapping APIs.

Exit criteria:
1. Mapping tables are centralized in a single native module.
2. Lowering/encoder no longer own divergent mapping copies.

### Phase 2.16: Typed internal ONNX stub model
Red:
1. Ensure ONNX encoder no longer traverses raw JSON everywhere.

Green:
1. Add typed internal ONNX model structs:
   - `graph_ir_core_onnx_model.hpp`
   - `graph_ir_core_onnx_model.cpp`
2. Convert ONNX stub JSON into typed model at encoder edge and encode from typed structs.

Exit criteria:
1. JSON boundary is isolated to parse edges for ONNX binary assembly.
2. Binary parity/oracle tests remain green.

### Phase 2.17: Style/layout parity cleanup
Red:
1. Normalize include/layout style for newly split files and headers.

Green:
1. Add internal encoder/lowering detail headers for explicit layering:
   - `graph_ir_core_onnx_encode_detail.hpp`
   - `graph_ir_core_lowering_internal.hpp`
2. Keep stable public boundary in `graph_ir_core.hpp` while constraining detail symbols to internal headers/TUs.

Exit criteria:
1. Internal layering is explicit and mlx-style focused.
2. New split units remain behavior-preserving.

### Phase 2.18: Validation sweep for split core
Red:
1. Re-run targeted and broad test gates to detect behavior drift.

Green:
1. Rebuild native extension with all split units.
2. Run targeted GraphIR + oracle tests and broad `test:fast[cpu]`.

Exit criteria:
1. All touched-path tests pass.
2. PRD/checklist updated with exact commands and outcomes.

### Phase 2.19: Remove six high-value duplication points
Red:
1. Identify the six remaining duplicated helper paths across core/native modules.
2. Add/keep behavior tests that fail on regressions while deduping.

Green:
1. Centralize shared helper logic in `detail/shared` modules.
2. Remove duplicate tagged-error and parsing/normalization helpers from local translation units.
3. Reuse shared native wrapper helper pipeline for repeated parse/export/encode flows.

Exit criteria:
1. No duplicate implementations remain for the six selected helper paths.
2. Behavior tests remain green.

### Phase 2.20: Folder-based graph_ir organization
Red:
1. Add/adjust boundary checks to require folder-based `graph_ir` module layout.
2. Capture failing signal if build/tooling still assumes flat `graph_ir_*` filenames.

Green:
1. Move `graph_ir` C++ code to subsystem folders (`graph_ir/core/...`, `graph_ir/core/lowering/...`, `graph_ir/core/onnx/...`).
2. Rewrite includes to folder paths.
3. Update extension build discovery/Makefile patching for recursive source+header trees.

Exit criteria:
1. Folder layout builds cleanly with no legacy flat include dependencies.
2. Targeted and broad GraphIR tests pass.

### Phase 3: Move to mlx repo + Python bindings
Red:
1. Add failing Python integration tests for GraphIR/ONNX paths.
2. Verify submodule build fails before GraphIR sources/bindings are wired.

Green:
1. Move GraphIR core sources into `mlx/mlx/graph_ir` as real files (no symlinks).
2. Build GraphIR core as part of `libmlx`; keep Python and Ruby wrappers thin and link to `libmlx`.
3. Add/update nanobind bindings for GraphIR/ONNX methods in `mlx/python/src`.
4. Add Python integration tests for GraphIR export/lowering/binary/compatibility flows.
5. Update Ruby extension build scripts to consume GraphIR core headers/symbols from `mlx/mlx/graph_ir`.

Exit criteria:
1. Editable Python package build succeeds with GraphIR core linked from `libmlx`.
2. New GraphIR Python tests pass.
3. Existing Python export/import tests remain green.
4. Ruby extension builds without compiling any duplicate GraphIR core sources under `ext/mlx`.

### Phase 3.1: Rename `graph_ir` subsystem to `ir` and flatten `core` path
Red:
1. Add/adjust boundary checks to fail when includes/build/tests still reference `graph_ir` paths or `ir/core/*` paths.
2. Capture failing signal for stale headers (`mlx/mlx/ir/core.hpp`, `mlx/mlx/ir/core/json.hpp`) after flatten.

Green:
1. Rename subsystem folders/files to `ir` across repo and `mlx` submodule.
2. Remove extra `core` directory segment and flatten to:
   - `mlx/mlx/ir.hpp`
   - `mlx/mlx/ir/json.hpp`
   - `mlx/mlx/ir/*`
3. Update build wiring/includes/tests/docs to new `ir` paths.
4. Keep runtime behavior and public six-method export flow unchanged.

Exit criteria:
1. No build/runtime path depends on `mlx/mlx/graph_ir` or `mlx/mlx/ir/core/*`.
2. Ruby `test/ir` gate passes with renamed layout.
3. Python GraphIR (`ir`) integration tests pass against editable install.

## Acceptance Criteria
1. Core/native split exists in `mlx-ruby` with clear boundary files.
2. Ruby GraphIR behavior remains compatible.
3. Extracted core module is organized for straightforward move into `mlx`.
4. Intermediate pre-upstream phases (2.5-2.12) are documented and sequenced before upstream merge.
5. Public API boundary, namespace/layout, file decomposition, mapping centralization, typed ONNX internal model, error conventions, and formatting are aligned with `mlx` style before Phase 3.
6. File/layout migration is complete with `ir` folder naming and no residual `core` path segment in the shared `mlx` core.

## Risks and Mitigations
1. Risk: split causes behavior drift in lowering/encoding output.
   Mitigation: retain existing test coverage and run targeted parity checks.
2. Risk: partial split leaves hidden Ruby coupling.
   Mitigation: keep core header free of `ruby.h` types and bind Ruby conversion only in wrapper file.
3. Risk: refactor introduces regressions.
   Mitigation: red/green per phase and rerun targeted graph_ir tests.

## Implementation Checklist
- [x] Phase 1 Red: Establish baseline with targeted GraphIR boundary/parity tests.
- [x] Phase 1 Green: Add `graph_ir_core.hpp/.cpp` and rewire Ruby entrypoints.
- [x] Phase 1 Exit: Run targeted GraphIR tests.
- [x] Phase 2 Red: Identify refactor mismatches vs mlx-style organization.
- [x] Phase 2 Green: Refactor core API/structure to mlx-style boundary.
- [x] Phase 2 Exit: Re-run targeted tests and confirm transplant-ready core.
- [x] Phase 2.5 Red: Record upstream-mlx style deltas (namespace/API/file layout/error/IO/tests).
- [x] Phase 2.5 Green: Finalize pre-migration style-alignment plan and concrete decomposition map.
- [x] Phase 2.5 Exit: Validate checklist against upstream `mlx` code patterns.
- [x] Phase 2.6 Red/Green/Exit: Align subsystem layout and namespace to upstream conventions.
- [x] Phase 2.7 Red/Green/Exit: Isolate JSON to edges and enforce typed public API boundary.
- [x] Phase 2.8 Red/Green/Exit: Split monolithic core into focused translation units.
- [x] Phase 2.9 Red/Green/Exit: Enforce public/internal header layering and detail hygiene.
- [x] Phase 2.10 Red/Green/Exit: Normalize error/diagnostic conventions.
- [x] Phase 2.11 Red/Green/Exit: Apply upstream style/format parity.
- [x] Phase 2.12 Red/Green/Exit: Finalize upstream Python parity test plan.
- [x] Phase 2.13 Red/Green/Exit: Land explicit lowering modular split (args/shape/ops/dispatch).
- [x] Phase 2.14 Red/Green/Exit: Split ONNX encoder into wire/tensor/assembly modules.
- [x] Phase 2.15 Red/Green/Exit: Centralize mapping/lookup tables.
- [x] Phase 2.16 Red/Green/Exit: Add typed internal ONNX model and JSON-edge conversion.
- [x] Phase 2.17 Red/Green/Exit: Complete style/layout layering cleanup for new modules.
- [x] Phase 2.18 Red/Green/Exit: Re-run targeted+broad validation and capture results.
- [x] Phase 2.19 Red/Green/Exit: Remove six high-value duplication points across core/native modules.
- [x] Phase 2.20 Red/Green/Exit: Complete folder-based graph_ir layout + recursive build integration.
- [x] Phase 3 Red/Green/Exit: Move GraphIR core into `mlx`, remove symlinking, and validate Python/Ruby bindings against shared `libmlx` core.
- [x] Phase 3.1 Red/Green/Exit: Rename `graph_ir` to `ir`, remove `core` path segment, and revalidate Ruby/Python gates.

## Phase 2.19-2.20 Execution Notes
1. Phase 2.19 duplication cleanup completed first:
   - Added shared helpers in `ext/mlx/graph_ir/core/detail.hpp` + `ext/mlx/graph_ir/core/shared.cpp`:
     - `tagged_error_message`
     - `normalize_integer_vector`
     - `parse_string_array`
     - `for_each_declared_payload_tensor`
   - Removed duplicated local helper implementations from:
     - `ext/mlx/graph_ir/core/api.cpp`
     - `ext/mlx/graph_ir/core/io.cpp`
     - `ext/mlx/graph_ir/core/lowering/ops.cpp`
     - `ext/mlx/graph_ir/core/onnx/model.cpp`
   - Reused shared helper flow in native wrappers (`ext/mlx/graph_ir/native.cpp`) to eliminate repeated parse/export/encode pipelines.
2. Phase 2.20 folder organization completed second:
   - Moved native/core files into:
     - `ext/mlx/graph_ir/native.cpp`
     - `ext/mlx/graph_ir/native.hpp`
     - `ext/mlx/graph_ir/core.hpp`
     - `ext/mlx/graph_ir/core/*`
     - `ext/mlx/graph_ir/core/lowering/*`
     - `ext/mlx/graph_ir/core/onnx/*`
   - Updated include paths to folder layout across moved files and `ext/mlx/native.cpp`.
   - Updated `ext/mlx/extconf.rb` to discover sources/headers recursively and patch `ORIG_SRCS`, `OBJS`, and `HDRS` in the generated Makefile.
   - Updated boundary test expectations in `test/graph_ir/graph_ir_core_boundary_style_test.rb`.
3. Validation (post Phase 2.20):
   - `bundle exec ruby ext/mlx/extconf.rb && make -C ext/mlx clean && make -C ext/mlx -j8`
   - `bundle exec ruby -Itest test/graph_ir/graph_ir_core_boundary_style_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/export_onnx_binary_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/export_onnx_compatibility_report_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/native_onnx_binary_python_oracle_test.rb`
   - `bundle exec rake test TEST="test/graph_ir/**/*_test.rb"` (CPU+GPU; each run: `162 runs, 1006 assertions, 0 failures, 0 errors, 11 skips`)

## Phase 3 Execution Notes
1. Replaced prior symlink baseline with real GraphIR core files in submodule:
   - added `mlx/mlx/graph_ir/core.hpp`
   - added `mlx/mlx/graph_ir/core/*`
   - added `mlx/mlx/graph_ir/core/lowering/*`
   - added `mlx/mlx/graph_ir/core/onnx/*`
2. Wired `libmlx` to own GraphIR core compilation:
   - updated `mlx/mlx/CMakeLists.txt` to compile GraphIR core translation units into `mlx`.
   - marked public GraphIR APIs with `MLX_API` in:
     - `mlx/mlx/graph_ir/core.hpp`
     - `mlx/mlx/graph_ir/core/json.hpp`
3. Updated Python bindings build to link against shared core instead of compiling duplicates:
   - removed GraphIR core `.cpp` entries from `mlx/python/src/CMakeLists.txt`.
   - kept `mlx/python/src/graph_ir.cpp` binding module and added JSON include compatibility for `<json.hpp>`.
4. Added nanobind GraphIR binding module:
   - new file `mlx/python/src/graph_ir.cpp`
   - registered from `mlx/python/src/mlx.cpp` via `init_graph_ir`.
5. Exposed Python public methods:
   - `export_graph_ir`
   - `export_graph_ir_json`
   - `export_onnx_compatibility_report`
   - `export_onnx_json`
   - `export_onnx`
   - `graph_ir_to_onnx_json`
   - `graph_ir_to_onnx`
   - `graph_ir_compatibility_report_json`
6. Added Python integration tests:
   - new file `mlx/python/tests/test_graph_ir.py`
7. Updated Ruby extension build integration to use submodule core only:
   - removed duplicated extension-owned core files:
     - `ext/mlx/graph_ir/core.hpp`
     - `ext/mlx/graph_ir/core/**/*`
   - kept only wrapper bindings in:
     - `ext/mlx/graph_ir/native.cpp`
     - `ext/mlx/graph_ir/native.hpp`
   - updated `ext/mlx/extconf.rb` to:
     - exclude `ext/mlx/graph_ir/core/**/*` from extension source/header compilation
     - force include/link paths against `mlx/mlx` headers and installed `libmlx`
     - patch generated Makefile include flags for `mlx/mlx` and JSON single-header compatibility
8. Removed temporary header-install exclusion that only existed for symlink packaging:
   - updated `mlx/CMakeLists.txt` to stop excluding `mlx/graph_ir/*` headers.
9. Updated boundary tests to assert new ownership location:
   - `test/graph_ir/graph_ir_core_boundary_style_test.rb` now validates files under `mlx/mlx/graph_ir`.
10. Validation (Phase 3):
   - `cd mlx && pip install --no-build-isolation -e . -v`
   - `cd mlx && DEVICE=cpu python -m unittest discover -v python/tests -p 'test_graph_ir.py'`
   - `cd mlx && DEVICE=cpu python -m unittest discover -v python/tests -p 'test_export_import.py'`
   - `bundle exec ruby ext/mlx/extconf.rb && make -C ext/mlx clean && make -C ext/mlx -j8`
   - `bundle exec ruby -Itest test/graph_ir/graph_ir_core_boundary_style_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/export_onnx_binary_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/export_onnx_compatibility_report_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/native_onnx_binary_python_oracle_test.rb`

## Phase 3.1 Execution Notes
1. Renamed repo-side GraphIR paths to `ir`:
   - `lib/mlx/graph_ir.rb` -> `lib/mlx/ir.rb`
   - `lib/mlx/graph_ir/webgpu_harness.rb` -> `lib/mlx/ir/webgpu_harness.rb`
   - `ext/mlx/graph_ir_native.cpp|hpp` -> `ext/mlx/ir/native.cpp|hpp`
   - `test/graph_ir/**/*` -> `test/ir/**/*`
2. Renamed submodule core paths and flattened `core` segment:
   - `mlx/mlx/graph_ir/core.hpp` -> `mlx/mlx/ir.hpp`
   - `mlx/mlx/graph_ir/core/*` -> `mlx/mlx/ir/*`
   - `mlx/mlx/graph_ir/core/lowering/*` -> `mlx/mlx/ir/lowering/*`
   - `mlx/mlx/graph_ir/core/onnx/*` -> `mlx/mlx/ir/onnx/*`
3. Updated build/binding integration:
   - `ext/mlx/native.cpp` include + init path updated to `ir/native.hpp`.
   - `mlx/python/src/mlx.cpp` initializes `init_ir`.
   - `mlx/python/src/CMakeLists.txt` compiles `ir.cpp`.
4. Removed stale extension build exclusions for deleted `ext/mlx/ir/core/*` in `ext/mlx/extconf.rb`.
5. Updated boundary test path assertions:
   - `test/ir/graph_ir_core_boundary_style_test.rb` now checks `mlx/mlx/ir.hpp` and `mlx/mlx/ir/json.hpp`.
6. Validation (Phase 3.1):
   - `bundle exec ruby ext/mlx/extconf.rb && make -C ext/mlx clean && make -C ext/mlx -j8`
   - `bundle exec ruby -Itest test/ir/graph_ir_core_boundary_style_test.rb`
   - `bundle exec ruby -Itest test/ir/export_onnx_binary_test.rb`
   - `bundle exec rake test TEST="test/ir/**/*_test.rb"`
   - `cd mlx && pip install --no-build-isolation -e . -v`
   - `cd mlx && DEVICE=cpu python -m unittest discover -v python/tests -p 'test_ir.py'`
   - `cd mlx && DEVICE=cpu python -m unittest discover -v python/tests -p 'test_export_import.py'`
   - `bundle exec rake test TEST="test/graph_ir/**/*_test.rb"` (CPU+GPU; each run: `162 runs, 1006 assertions, 0 failures, 0 errors, 11 skips`)

## Phase 1-2 Execution Notes
1. Added native core boundary files:
   - `ext/mlx/graph_ir_core.hpp`
   - `ext/mlx/graph_ir_core_lowering.cpp`
   - `ext/mlx/graph_ir_core_onnx_encode.cpp`
2. Moved ONNX lowering, compatibility reporting, protobuf encoding, and binary IO into core namespace (`mlx::core::graph_ir`).
3. Kept Ruby-facing parsing/capture/exception translation in `ext/mlx/graph_ir_native.cpp`.
4. Updated native wrappers to call core methods without Ruby intermediates for ONNX lowering/encoding paths.
5. Updated `ext/mlx/extconf.rb` Makefile source patching so the extension build includes the decomposed core units (`graph_ir_core_api.cpp`, `graph_ir_core_compat.cpp`, `graph_ir_core_io.cpp`, `graph_ir_core_lowering.cpp`, `graph_ir_core_onnx_encode.cpp`, `graph_ir_core_shared.cpp`).
6. Refactored extracted core organization to explicit subsystem namespace layering (`mlx::core::graph_ir` + internal `detail`).

## Phase 2.5-2.12 Execution Notes
1. Phase 2.5 (delta baseline): compared extracted GraphIR core against upstream `mlx/mlx` conventions for namespace depth, header boundaries, file decomposition, diagnostics, and formatting.
2. Phase 2.6 (namespace/layout): finalized core symbol namespace as `mlx::core::graph_ir` and mapped files for direct upstream relocation into a `mlx/mlx/graph_ir/*` layout.
3. Phase 2.7 (typed boundary + JSON edge isolation): kept `ext/mlx/graph_ir_core.hpp` free of JSON types, retained typed boundary structs (`OnnxBinaryWriteOptions`, `OnnxBinaryArtifact`), and isolated `OrderedJson` APIs to `ext/mlx/graph_ir_core_json.hpp`.
4. Phase 2.8 (decomposition): split core implementation into focused translation units:
   - `ext/mlx/graph_ir_core_lowering.cpp` (lowering + compatibility internals)
   - `ext/mlx/graph_ir_core_onnx_encode.cpp` (ONNX protobuf encoder + binary artifact build internals)
   - `ext/mlx/graph_ir_core_api.cpp` (string boundary adapters)
   - `ext/mlx/graph_ir_core_io.cpp` (artifact file write + unsupported-message classification)
5. Phase 2.9 (header layering): enforced public/internal separation with:
   - public stable boundary: `ext/mlx/graph_ir_core.hpp`
   - internal JSON detail boundary: `ext/mlx/graph_ir_core_json.hpp`
6. Phase 2.10 (diagnostics): normalized tags to `[graph_ir.lowering]`, `[graph_ir.api]`, and `[graph_ir.io]`; preserved unsupported-op promotion by accepting both new and legacy unsupported prefixes in `graph_ir_is_unsupported_error_message`.
7. Phase 2.11 (format/style): applied upstream `.clang-format` to all extracted core files (`graph_ir_core.hpp`, `graph_ir_core_json.hpp`, `graph_ir_core_api.cpp`, `graph_ir_core_compat.cpp`, `graph_ir_core_io.cpp`, `graph_ir_core_lowering.cpp`, `graph_ir_core_onnx_encode.cpp`, `graph_ir_core_shared.cpp`).
8. Phase 2.12 (upstream parity planning): finalized the method-by-method Python parity matrix below and tied each method to existing ruby-side parity/oracle tests for porting.

## Phase 2.13-2.18 Execution Notes
1. Added a red boundary test in `test/graph_ir/graph_ir_core_boundary_style_test.rb` to require concrete split module files; observed failing signal before implementation.
2. Landed lowering-focused split units and dispatch boundary:
   - `ext/mlx/graph_ir_core_lowering_args.cpp`
   - `ext/mlx/graph_ir_core_lowering_shape.cpp`
   - `ext/mlx/graph_ir_core_lowering_ops.cpp`
   - `ext/mlx/graph_ir_core_lowering_dispatch.cpp`
   - `ext/mlx/graph_ir_core_lowering_internal.hpp`
3. Landed ONNX encoder decomposition:
   - `ext/mlx/graph_ir_core_onnx_wire.cpp`
   - `ext/mlx/graph_ir_core_onnx_tensor.cpp`
   - `ext/mlx/graph_ir_core_onnx_assemble.cpp`
   - `ext/mlx/graph_ir_core_onnx_encode_detail.hpp`
4. Added typed ONNX model conversion layer:
   - `ext/mlx/graph_ir_core_onnx_model.hpp`
   - `ext/mlx/graph_ir_core_onnx_model.cpp`
5. Centralized mapping/lookup tables:
   - `ext/mlx/graph_ir_core_mappings.hpp`
   - `ext/mlx/graph_ir_core_mappings.cpp`
6. Updated lowering and ONNX encode paths to consume shared mapping APIs and typed ONNX model conversion at JSON edges.
7. Validation commands:
   - `bundle exec ruby ext/mlx/extconf.rb && make -C ext/mlx clean && make -C ext/mlx`
   - `bundle exec ruby -Itest test/graph_ir/graph_ir_core_boundary_style_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/export_onnx_binary_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/export_onnx_compatibility_report_test.rb`
   - `bundle exec ruby -Itest test/graph_ir/native_onnx_binary_python_oracle_test.rb`
   - `bundle exec rake "test:fast[cpu]"`
8. Validation outcomes:
   - Targeted suites passed (`0 failures`, `0 errors`).
   - Broad suite passed (`896 runs, 349736 assertions, 0 failures, 0 errors, 31 skips`).

## Post-2.12 High-Value Extraction Execution
1. Added internal detail boundary header `ext/mlx/graph_ir_core_detail.hpp` for shared helper APIs and internal lowering structs used across translation units.
2. Moved shared numeric/dtype normalization helpers into `ext/mlx/graph_ir_core_shared.cpp`:
   - `json_is_numeric`
   - `normalized_integer_scalar`
   - `canonical_dtype`
   - `onnx_effective_dtype`
   - `onnx_dtype_symbol`
3. Moved compatibility reporting implementation into dedicated unit `ext/mlx/graph_ir_core_compat.cpp`:
   - `graph_ir_compatibility_report_payload_impl`
4. Kept lowering orchestration in `ext/mlx/graph_ir_core_lowering.cpp` and ONNX protobuf/tensor/model encoding in `ext/mlx/graph_ir_core_onnx_encode.cpp`, with cross-unit linkage through `detail` declarations.
5. Updated generated extension build sources to compile all units:
   - `graph_ir_core_api.cpp`
   - `graph_ir_core_compat.cpp`
   - `graph_ir_core_io.cpp`
   - `graph_ir_core_lowering.cpp`
   - `graph_ir_core_onnx_encode.cpp`
   - `graph_ir_core_shared.cpp`

## Post-2.12 High-Value Extraction Validation
1. `make -C ext/mlx clean && make -C ext/mlx`
2. `bundle exec ruby -Itest test/graph_ir/graph_ir_core_boundary_style_test.rb`
3. `bundle exec ruby -Itest test/graph_ir/graph_ir_export_contract_boundary_test.rb`
4. `bundle exec ruby -Itest test/graph_ir/export_onnx_binary_test.rb`
5. `bundle exec ruby -Itest test/graph_ir/export_onnx_compatibility_report_test.rb`
6. `bundle exec ruby -Itest test/graph_ir/native_onnx_binary_python_oracle_test.rb`
7. `bundle exec ruby -Itest test/graph_ir_native_timing_test.rb`
8. `bundle exec rake "test:fast[cpu]"` (895 runs, 349716 assertions, 0 failures, 0 errors, 31 skips)

## Phase 2.12 Parity Matrix (for Phase 3 Port)
1. `export_onnx(target_path, fun, args...)`
   - Ruby parity anchors: `test/graph_ir/export_onnx_binary_test.rb`, `test/graph_ir/export_onnx_direct_test.rb`, `test/graph_ir/export_onnx_external_data_test.rb`
   - Python phase-3 port: validate byte-level ONNX equivalence and external-data artifact behavior.
2. `export_onnx_json(fun, args...)`
   - Ruby parity anchors: `test/graph_ir/export_onnx_shapeless_facade_test.rb`, `test/graph_ir/graph_ir_native_timing_test.rb`
   - Python phase-3 port: validate deterministic JSON stub shape and timing probe schema.
3. `export_graph_ir(fun, args...)`
   - Ruby parity anchors: `test/graph_ir/graph_ir_export_contract_boundary_test.rb`
   - Python phase-3 port: validate hash/dict contract keys and tensor metadata shape.
4. `export_graph_ir_json(fun, args...)`
   - Ruby parity anchors: `test/graph_ir/graph_ir_export_contract_boundary_test.rb`, `test/graph_ir/graph_ir_serialized_sources_test.rb`
   - Python phase-3 port: validate JSON payload determinism and source serialization compatibility.
5. `graph_ir_to_onnx(target_path, graph_ir_source, ...)`
   - Ruby parity anchors: `test/graph_ir/export_onnx_direct_test.rb`, `test/graph_ir/export_onnx_binary_test.rb`
   - Python phase-3 port: validate source-to-binary path with no Ruby-layer intermediates.
6. `graph_ir_to_onnx_json(graph_ir_source, ...)`
   - Ruby parity anchors: `test/graph_ir/graph_ir_onnx_stub_test.rb`, `test/graph_ir/graph_ir_serialized_sources_test.rb`
   - Python phase-3 port: validate source-to-JSON behavior for Hash/JSON-string/file/IO-like sources.
7. Cross-method binary oracle parity
   - Ruby anchor: `test/graph_ir/native_onnx_binary_python_oracle_test.rb`
   - Python phase-3 port: keep this as the canonical byte-level oracle between Python builder and core encoder.

## Phase 1-2 Validation
1. `make -C ext/mlx`
2. `bundle exec ruby -Itest test/graph_ir/graph_ir_export_contract_boundary_test.rb`
3. `bundle exec ruby -Itest test/graph_ir/export_onnx_binary_test.rb`
4. `bundle exec ruby -Itest test/graph_ir/export_onnx_compatibility_report_test.rb`
5. `bundle exec ruby -Itest test/graph_ir/native_onnx_binary_python_oracle_test.rb`
6. `bundle exec ruby -Itest test/graph_ir_native_timing_test.rb`
7. `bundle exec ruby -Itest test/graph_ir/graph_ir_core_boundary_style_test.rb`
8. `bundle exec rake "test:fast[cpu]"` (895 runs, 349716 assertions, 0 failures, 0 errors, 31 skips)

## Phase 3.2 Execution Notes
1. Consolidated all lowering translation units from `mlx/mlx/ir/lowering/*` into `mlx/mlx/ir/lowering.cpp`.
2. Consolidated all ONNX translation units from `mlx/mlx/ir/onnx/*` into `mlx/mlx/ir/onnx.cpp`.
3. Removed old foldered lowering/onnx source/header files and updated `mlx/mlx/CMakeLists.txt` to compile only:
   - `mlx/mlx/ir/lowering.cpp`
   - `mlx/mlx/ir/onnx.cpp`
4. Updated boundary expectations in `test/ir/graph_ir_core_boundary_style_test.rb`.
5. Validation:
   - `bundle exec ruby ext/mlx/extconf.rb && make -C ext/mlx clean && make -C ext/mlx -j8`
   - `bundle exec ruby -Itest test/ir/graph_ir_core_boundary_style_test.rb`
   - `bundle exec ruby -Itest test/ir/export_onnx_binary_test.rb`
   - `bundle exec rake test TEST="test/ir/**/*_test.rb"` (`162 runs, 994 assertions, 0 failures, 0 errors, 11 skips` on CPU and GPU)
   - `cd mlx && pip install --no-build-isolation -e . -v`
   - `cd mlx && DEVICE=cpu python -m unittest discover -v python/tests -p 'test_ir.py'`
   - `cd mlx && DEVICE=cpu python -m unittest discover -v python/tests -p 'test_export_import.py'`
