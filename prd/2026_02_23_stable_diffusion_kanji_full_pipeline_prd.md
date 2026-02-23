# 2026_02_23 Stable Diffusion Kanji Full Pipeline PRD

## Status
Completed (2026-02-23)

## Context
The Stable Diffusion web demo needed to move from tiny/fallback behavior to real pretrained text encoder + UNet + VAE exports generated natively in Ruby, with no external ONNX files. After enabling full models, browser session initialization failed due runtime/memory constraints, and the demo naming needed to be updated to Stable Diffusion Kanji.

## Goals
1. Export real pretrained Stable Diffusion components (text encoder, UNet, VAE decoder) via native MLX GraphIR ONNX export.
2. Keep a Ruby-native path for HF weight loading and conversion fallback (no Python ONNX export path).
3. Make browser runtime initialization stable for the full pipeline on the web demo.
4. Rename web-facing demo and asset model identity to Stable Diffusion Kanji.

## Non-goals
1. Introducing external pre-exported ONNX artifacts.
2. Reintroducing legacy TinyUnet/fallback export contracts.
3. Changing the public web route structure (`/demo/stable_diffusion/`).

## Phased Plan (Red/Green)

### Phase 1: Red runtime capture
Red:
1. Run `tasks/web_assets_task/export_stable_diffusion_assets.rb` after TinyUnet removal and capture the breakage.
2. Run browser/integration probe and capture ONNX session initialization failures.

Green:
1. Confirm failure signals:
- exporter NameError on `TinyUnetModel`
- browser session creation failures for VAE/text-unet combinations.

Exit criteria:
1. Reproducible failing signals recorded for both exporter and browser initialization.

### Phase 2: Green exporter/model migration
Red:
1. Keep failing path while replacing Tiny-specific exporter wiring.

Green:
1. Replace exporter flow with full-model loaders:
- `StableDiffusionExample.download_hf_weights!`
- `load_text_encoder_from_hf_directory`
- `load_unet_from_hf_directory`
- `load_autoencoder_from_hf_directory`
2. Export all three ONNX binaries natively via `MLX::GraphIR.export_onnx`.
3. Remove tiny subset/fallback codepaths from SD web asset export task.
4. Fix large NPZ fallback loading reliability by materializing tensors during NPZ extraction load.

Exit criteria:
1. `bundle exec ruby tasks/web_assets_task/export_stable_diffusion_assets.rb` completes successfully.
2. `web/assets/stable_diffusion/meta.json` references full-pipeline metadata.

### Phase 3: Green runtime stabilization
Red:
1. Probe browser load failures for full-model ONNX assets.

Green:
1. Add float16 tensor encode/decode handling in `web/demo/stable_diffusion/main.js`.
2. Prioritize VAE provider initialization with `webgpu` first.
3. Configure SD export runtime footprint for browser stability:
- text/unet dtype: float16
- vae dtype: float32
- latent sample size: 32
4. Re-export assets and verify page can fully initialize and generate.

Exit criteria:
1. Stable Diffusion web page reaches ready state with providers loaded.
2. Stable Diffusion integration test passes.

### Phase 4: Green rename to Kanji
Red:
1. Identify all web/UI/asset name references still using Tiny naming.

Green:
1. Rename web-facing SD strings to Stable Diffusion Kanji in:
- `web/index.html`
- `web/demo/stable_diffusion/index.html`
- web asset model name prefix via exporter metadata.
2. Update related wiring/integration expectations.

Exit criteria:
1. Demo card/title/model status use Kanji naming consistently.

## Acceptance Criteria
1. SD web assets are generated from full pretrained components without external ONNX downloads.
2. Browser demo initializes sessions and enables generation.
3. Stable Diffusion demo name is Stable Diffusion Kanji in web-facing surfaces and metadata model identity.
4. Targeted unit/task tests and all three web integration tests pass.

## Risks and Mitigations
1. Risk: Browser OOM/session failures with large ONNX models.
   Mitigation: mixed precision exports and reduced latent sample size (32) for web runtime.
2. Risk: safetensors-unavailable environments fail loading large fallback NPZ files.
   Mitigation: materialized NPZ loading path to avoid mmap/file-handle exhaustion.
3. Risk: naming drift between metadata and UI/tests.
   Mitigation: update and validate wiring + integration expectations.

## Implementation Checklist
- [x] Phase 1 Red: Capture exporter and browser runtime failures.
- [x] Phase 2 Green: Replace Tiny exporter path with full-model loaders and native ONNX export.
- [x] Phase 2 Green: Fix large NPZ fallback load reliability.
- [x] Phase 2 Exit: SD asset export command succeeds locally.
- [x] Phase 3 Green: Add float16 runtime tensor support in SD web demo.
- [x] Phase 3 Green: Stabilize provider/runtime footprint for browser load.
- [x] Phase 3 Exit: Stable Diffusion integration test passes.
- [x] Phase 4 Green: Rename SD web/demo/assets naming to Stable Diffusion Kanji.
- [x] Validation: Run targeted task/unit tests and all 3 web integration tests.
