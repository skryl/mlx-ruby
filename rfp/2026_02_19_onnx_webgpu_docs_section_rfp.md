# Onnx/WebGPU Support Docs Section RFP

## Status

Completed (2026-02-19)

## Context

The current docs include ONNX/WebGPU material inside `ruby/export.rst`, but it
is not presented as a dedicated top-level sidebar section with a multi-page
guide. Users need an explicit navigation section named `Onnx/WebGPU Support`
with clear, sequential docs for the full MLX -> Graph IR -> ONNX -> WebGPU
path, including examples.

## Goals

1. Add a dedicated sidebar caption: `Onnx/WebGPU Support`.
2. Add multiple focused docs pages covering the complete export/runtime flow.
3. Include runnable command/snippet examples for each stage.
4. Keep existing `ruby/export.rst` API docs compatible while improving
   discoverability via the new section.

## Non-goals

1. Changing exporter/runtime implementation behavior.
2. Adding new ONNX operators or runtime features.
3. Rewriting all existing export API reference content.

## Phased Plan (Red/Green)

### Phase 1: Sidebar section scaffold

Red:
1. Run a reproducible check showing the section is absent:
   - `rg -n ":caption: Onnx/WebGPU Support" docs/src/index.rst`
2. Capture failing signal (`exit 1` / no matches).

Green:
1. Add new toctree in `docs/src/index.rst` with caption
   `Onnx/WebGPU Support`.
2. Wire placeholder page entries under the new section.

Refactor:
1. Keep surrounding toctree ordering coherent and minimal.

Exit criteria:
1. `docs/src/index.rst` contains the new caption and page links.

### Phase 2: Multi-page process docs with examples

Red:
1. Run a reproducible check showing the target page set is missing:
   - `find docs/src/onnx_webgpu -maxdepth 1 -type f`
2. Capture failing signal (missing directory/pages).

Green:
1. Create docs pages under `docs/src/onnx_webgpu/` covering:
   - Overview and navigation
   - MLX -> Graph IR export
   - Graph IR validation + WebGPU compatibility preflight
   - Graph IR -> ONNX export/stub path
   - WebGPU harness packaging + smoke validation
   - End-to-end examples/workflows
2. Add concrete snippet-based examples for each phase.

Refactor:
1. Reduce duplicate wording between new pages and `ruby/export.rst`.
2. Add cross-links between API reference and process docs.

Exit criteria:
1. All planned pages exist and are linked from the new sidebar section.
2. Each page includes at least one code/command example.

### Phase 3: Build verification and nav validation

Red:
1. Rebuild docs and confirm nav/page references fail before changes (baseline
   already captured in prior user report).

Green:
1. Run `bundle exec rake docs:build`.
2. Verify generated sidebar contains `Onnx/WebGPU Support` and child pages.
3. Verify generated pages include ONNX/WebGPU process content.

Refactor:
1. Fix obvious heading/link consistency issues found during build validation.

Exit criteria:
1. Docs build succeeds.
2. Generated sidebar exposes the new section and page set.

## Acceptance Criteria

1. `Onnx/WebGPU Support` appears as a dedicated sidebar section in generated
   docs.
2. Section contains multiple pages documenting the full MLX -> Graph IR ->
   ONNX -> WebGPU process.
3. Pages contain practical examples and clear stage-by-stage guidance.
4. Docs build completes successfully.

## Risks and Mitigations

1. Risk: Navigation clutter/duplication.
   Mitigation: Keep API reference in `ruby/export.rst` and focus new section on
   workflow guidance; cross-link instead of copy-pasting.
2. Risk: Drift between docs and implementation names.
   Mitigation: Reuse function/task names already documented in
   `docs/src/ruby/export.rst`.
3. Risk: Build regressions from RST formatting.
   Mitigation: run full `bundle exec rake docs:build` and inspect warnings.

## Implementation Checklist

- [x] Phase 1 red check captured (caption absent before change).
- [x] Phase 1 green: add `Onnx/WebGPU Support` toctree section.
- [x] Phase 2 red check captured (page set absent before change).
- [x] Phase 2 green: create multi-page `docs/src/onnx_webgpu/*`.
- [x] Phase 2 refactor: add cross-links and remove obvious duplication.
- [x] Phase 3 green: build docs and verify generated sidebar/pages.
- [x] Finalize status to `Completed` with date once all criteria are met.
