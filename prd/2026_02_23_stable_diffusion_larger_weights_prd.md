# 2026_02_23 Stable Diffusion Larger Weights PRD

## Status
Completed (2026-02-23)

## Context
The current Stable Diffusion web demo uses `Narsil/tiny-stable-diffusion-torch` weights. The resulting outputs are frequently too bland and do not demonstrate meaningful prompt-conditioned behavior.

We need a larger pretrained checkpoint source while keeping:

1. Native ONNX export path (no Python ONNX builder).
2. Existing TinyUnet web demo architecture/runtime contract.
3. Existing no-safetensors fallback behavior in environments where `MLX::Core.load` cannot load `.safetensors` directly.

## Goals
1. Switch default Stable Diffusion HF weights to a meaningfully larger pretrained checkpoint.
2. Keep exporter/runtime flow stable for web demos and `rake web:assets`.
3. Make TinyUnet HF mapping robust for larger source tensor shapes via explicit slicing.
4. Preserve no-safetensors compatibility using a preconverted subset fallback file.

## Non-goals
1. Replacing TinyUnet demo architecture with full Stable Diffusion UNet.
2. Reintroducing Python ONNX export paths.
3. Adding new demo pages or scheduler algorithms.

## Phased Plan (Red/Green)

### Phase 1: Red tests for larger-checkpoint mapping

Red:
1. Add failing parity test asserting TinyUnet can load a larger-shaped HF state dict by slicing to local tensor shapes.
2. Add failing task/guardrail test locking the new default Stable Diffusion HF repo id.

Green:
1. Confirm tests fail against current mapping/default repo.

Exit criteria:
1. Failure signals are captured for both mapping and default-repo contract.

### Phase 2: Green implementation

Red:
1. Keep new tests failing while code still uses tiny-only assumptions.

Green:
1. Update `StableDiffusionExample::HF_REPO_ID` to the larger checkpoint.
2. Update TinyUnet HF mapping to slice conv/bias tensors to expected TinyUnet dimensions.
3. Add preconverted `.mlx_subset.npz` fallback artifact for the new default repo.
4. Update Stable Diffusion web exporter to copy fallback subset artifact into the downloaded weights directory when needed.

Exit criteria:
1. New red tests pass.
2. `bundle exec rake web:assets` succeeds in this environment without requiring native safetensors support.

### Phase 3: Integration validation

Red:
1. Run web integration checks and capture any breakage due to larger weights.

Green:
1. Regenerate web assets with new default repo.
2. Run targeted wiring/guardrail/parity tests and three web integration tests.

Exit criteria:
1. Stable Diffusion demo loads and generates in browser integration test.
2. GPT-2 and nanoGPT integration remain green (regression gate).

## Acceptance Criteria
1. Default Stable Diffusion weights repo is updated to a larger checkpoint.
2. TinyUnet mapping supports larger source tensor shapes with deterministic slicing.
3. `rake web:assets` succeeds on this machine (including no-safetensors fallback path).
4. Stable Diffusion integration test passes with regenerated assets.
5. No regressions in GPT-2/nanoGPT integration tests.

## Risks and Mitigations
1. Risk: larger checkpoint key/layout drift from expected names.
   Mitigation: mapping test covers large-shape load behavior and key presence assumptions.
2. Risk: environments without native safetensors support fail.
   Mitigation: checked-in `.mlx_subset.npz` fallback copied by exporter.
3. Risk: browser latency rises with larger latent shapes.
   Mitigation: keep current integration step count low and verify runtime remains functional.

## Implementation Checklist
- [x] Phase 1 Red: Add failing large-shape mapping test.
- [x] Phase 1 Red: Add failing default-repo guardrail assertion.
- [x] Phase 1 Green: Capture red failure signals.
- [x] Phase 2 Green: Update default repo constant.
- [x] Phase 2 Green: Implement conv/bias slicing in TinyUnet mapper.
- [x] Phase 2 Green: Add and wire preconverted fallback subset file.
- [x] Phase 2 Exit: `bundle exec rake web:assets` succeeds locally.
- [x] Phase 3 Green: Run targeted parity/wiring/guardrail tests.
- [x] Phase 3 Green: Run all 3 web integration tests.
- [x] Mark PRD Completed with date after all phases pass.
