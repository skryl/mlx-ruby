# 2026_02_19 Stable Diffusion Pretrained Weights RFP

## Status

Completed (2026-02-19)

## Context

The current Stable Diffusion web demo assets are generated from a synthetic `mlx-ruby-examples` pipeline, which does not load fully pretrained diffusion/text/decoder checkpoints. This causes poor image quality and does not satisfy the requirement to rely on pretrained weights.

Requested behavior:

1. Use pretrained model weights (Hugging Face checkpoints).
2. Do not consume externally pre-exported ONNX artifacts.
3. Export ONNX locally from downloaded pretrained weights during the repo asset pipeline.

## Goals

1. Replace the Stable Diffusion asset exporter with a local ONNX export flow from pretrained HF weights.
2. Keep the output contract used by web demo (`text_encoder.onnx`, `unet.onnx`, `vae_decoder.onnx`, `model.onnx`, metadata/json configs).
3. Update demo runtime to match exported tensor dtypes/shapes.
4. Add guardrail tests for `weights-only` and `no external ONNX` policy.

## Non-goals

1. Introducing new model architectures beyond current tiny SD demo target.
2. Shipping a full training pipeline for Stable Diffusion.
3. Implementing advanced schedulers beyond current demo baseline.

## Phased Plan (Red/Green)

### Phase 1: Guardrails and failing checks

Red:
1. Update SD guardrail tests to require local export from pretrained weights.
2. Add failing assertions to reject synthetic pipeline dependency and external ONNX download flow.

Green:
1. Ensure tests reflect intended behavior and fail against current implementation.

Exit criteria:
1. Targeted test run shows failure before implementation.

### Phase 2: Exporter rewrite to local ONNX from pretrained weights

Red:
1. Keep Phase 1 tests failing while exporter still points at synthetic path.

Green:
1. Rewrite `tasks/web_assets_task/export_stable_diffusion_assets.rb` to:
   - fetch pretrained weights,
   - export ONNX locally via Python runtime,
   - write metadata/config files for the web demo.
2. Keep compatibility alias `model.onnx -> unet.onnx`.

Exit criteria:
1. Exporter tests pass.
2. `rake web:assets` produces SD assets from local export pipeline.

### Phase 3: Web demo runtime alignment

Red:
1. Verify runtime assumptions mismatch exported I/O types/shapes.

Green:
1. Update SD demo JS to feed correct tensor dtypes/layout for exported models.
2. Preserve multistep denoise UX and output rendering.

Exit criteria:
1. Wiring and guardrail tests pass.
2. Demo initializes on generated assets without dtype mismatch errors.

## Acceptance Criteria

1. SD exporter uses pretrained HF weights and local ONNX export.
2. SD exporter does not download any ONNX file from external sources.
3. SD web demo references generated local ONNX assets and runs with correct dtypes.
4. Targeted task tests pass.

## Risks and Mitigations

1. Risk: local Python dependency mismatch (`diffusers`, `transformers`, `torch`) can break export.
   Mitigation: add clear failure diagnostics and environment variable overrides for python bin/repo id.
2. Risk: ONNX input dtypes differ from prior demo assumptions.
   Mitigation: metadata-driven input handling and explicit tensor type conversion in JS.
3. Risk: export time/size overhead.
   Mitigation: keep tiny SD default model and reuse output directory.

## Implementation Checklist

- [x] Phase 1 red checks added and failing signal captured.
- [x] Phase 1 exit criteria met.
- [x] Phase 2 green implementation complete.
- [x] Phase 2 exit criteria met.
- [x] Phase 3 green implementation complete.
- [x] Phase 3 exit criteria met.
