# 2026_02_19 Web Demo Weights Required RFP

## Status

Completed (2026-02-19)

## Context

Web demos currently allow random-init fallback paths (notably nanoGPT exports) and UI states that do not clearly block generation when usable weights are missing.

The requested behavior is:

1. Remove random-init fallback for demo export paths.
2. If weights are missing, show that in the `Weights:` badge.
3. Generation actions remain disabled when weights are missing.

## Goals

1. Remove random-init export fallback from nanoGPT web asset exporters.
2. Ensure GPT-2 and nanoGPT demos disable `Generate` until usable weights/session are available.
3. Ensure CVAE demo also surfaces a `Weights:` badge and disables interactive inference controls when model weights are unavailable.
4. Add targeted tests to lock this behavior.

## Non-goals

1. Reworking model architectures or tokenizer semantics.
2. Changing training algorithms or hyperparameters.
3. Adding new weight distribution channels.

## Phased Plan (Red/Green)

### Phase 1: Demo UI gating + messaging

Red:
1. Add failing tests asserting:
   - language demos include missing-weights messaging and disabled generate path,
   - CVAE includes a `Weights:` badge and missing-weights path.

Green:
1. Update demo JS/HTML to:
   - default generation/interactive controls to disabled,
   - detect missing weights/model assets before session init,
   - keep actions disabled and set `Weights:` badge to missing when unavailable.

Exit criteria:
1. New UI behavior tests pass.
2. Existing nanoGPT wiring test remains green.

### Phase 2: Exporter fallback removal

Red:
1. Failing tests assert nanoGPT exporter no longer advertises or uses random-init export mode.

Green:
1. Update nanoGPT exporter to require trained artifacts and skip random-init export behavior.
2. Update docs to reflect required training artifacts for these demos.

Exit criteria:
1. Exporter fallback tests pass.
2. Exporter source no longer emits random-init mode path.

## Acceptance Criteria

1. `web/demo/gpt2/main.js` and `web/demo/nanogpt/main.js` keep `Generate` disabled when weights are missing.
2. `Weights:` badges explicitly show missing/unavailable weight status instead of implying random-init fallback.
3. `tasks/web_assets_task/export_nanogpt_assets.rb` no longer falls back to random-init export.
4. CVAE demo UI surfaces a `Weights:` badge and disables inference controls if weights are unavailable.
5. Targeted task tests pass.

## Risks and Mitigations

1. Risk: stricter gating blocks generation due stale/missing metadata.
   Mitigation: rely on model asset existence checks and explicit status text.
2. Risk: exporter behavior change surprises users who relied on random-init demos.
   Mitigation: update docs with required train/export flow and clear missing-weights UI.
3. Risk: regression in existing nanoGPT route behavior.
   Mitigation: keep and run existing nanoGPT wiring test.

## Implementation Checklist

- [x] Phase 1 red check captured.
- [x] Phase 1 green implementation complete.
- [x] Phase 1 exit criteria met.
- [x] Phase 2 red check captured.
- [x] Phase 2 green implementation complete.
- [x] Phase 2 exit criteria met.
