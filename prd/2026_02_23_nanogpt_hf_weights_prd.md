# 2026_02_23 nanoGPT HF Weights PRD

## Status
Completed (2026-02-23)

## Context
The nanoGPT web asset flow depended on locally trained artifacts (`weights.npz`, `training_config.json`) and skipped export when those files were absent. We need a reproducible, upstream-friendly path that mirrors GPT-2: fetch trained checkpoint weights from Hugging Face, load them natively in Ruby/MLX, and export ONNX without local training prerequisites.

## Goals
1. Load trained nanoGPT Shakespeare checkpoint weights from Hugging Face in native Ruby/MLX flow.
2. Make `tasks/web_assets_task/export_nanogpt_assets.rb` export from HF weights every run (no local-training gate).
3. Keep ONNX export native (`MLX::GraphIR.export_onnx`) with no Python fallback in exporter path.
4. Preserve web demo readiness and generation behavior for nanoGPT.
5. Add/adjust tests to lock HF mapping behavior and web integration behavior.

## Non-goals
1. Replacing the standalone nanoGPT training task (`web:train[nanogpt]`) for users who still want local training experiments.
2. Changing GPT-2 or Stable Diffusion export model sources.
3. Building a generalized tokenizer downloader for all custom char-level models.

## Phased Plan (Red/Green)

### Phase 1: HF nanoGPT model loading primitives
Red:
1. Add failing parity tests for `NanoGptExample::Config.from_hf_config` and HF state-dict mapping/loading.
2. Add failing fallback test for safetensors->npz load path in `NanoGptExample::NanoGptModel.from_hf_directory`.

Green:
1. Add HF config struct and downloader/loaders to `examples/web/nanogpt_example.rb`.
2. Implement HF state mapping for nanoGPT blocks (q/k/v split, MLP, embeddings, layer norm, lm_head).
3. Support both Conv1D-style and Linear-style tensor layouts for checkpoint compatibility.
4. Add safetensors-native-unavailable fallback to `.npz` conversion.

Refactor:
1. Keep helper methods private and co-locate mapping utilities near model class.

Exit criteria:
1. New parity test file passes locally.
2. HF load path can instantiate a model from `config.json + model.safetensors`.

### Phase 2: Web exporter migration to HF weights
Red:
1. Update guardrail tests to fail until exporter references HF repo flow and no longer references local `weights.npz` dependency.

Green:
1. Rewrite `tasks/web_assets_task/export_nanogpt_assets.rb` to:
   - fetch HF weights to `web/assets/nanogpt/weights`
   - cache repo id marker
   - build tokenizer from dataset text
   - export ONNX natively
   - write metadata/tokenizer/preset outputs
2. Keep output metadata aligned with demo consumption and include HF provenance.
3. Update demo missing-assets message to point to `bundle exec rake web:assets`.
4. Update docs/README statements about nanoGPT export prerequisites.

Refactor:
1. Keep exporter helper methods consistent with GPT-2 exporter patterns (dtype parsing, marker handling, timing blocks).

Exit criteria:
1. Exporter runs successfully without pre-existing local training artifacts.
2. `web/assets/nanogpt/model.onnx` and `meta.json` are regenerated from HF weights.

### Phase 3: Integration hardening
Red:
1. Integration probe fails if generation validation is too strict for whitespace/newline token outputs.

Green:
1. Update integration probe validation to accept generated token IDs as valid generation evidence when rendered text is empty.
2. Run integration tests for GPT-2, nanoGPT, and Stable Diffusion demos.

Refactor:
1. Keep validation semantics strict on generation activity (timing + top-k + text-or-ids).

Exit criteria:
1. All three web integration tests pass.
2. No regressions in related wiring/guardrail/task tests.

## Acceptance Criteria (Full Completion)
1. nanoGPT exporter no longer depends on local `weights.npz`/`training_config.json`.
2. nanoGPT HF checkpoint (`sosier/nanoGPT-shakespeare-char-weights-not-tied`) is the default source.
3. ONNX export remains native-only from Ruby path.
4. Demo integration tests pass for all three model pages.
5. Added parity tests cover HF mapping and safetensors fallback behavior.

## Risks and Mitigations
1. Risk: HF tensor layout mismatch (Conv1D vs Linear) causes shape/value errors.
   Mitigation: layout-aware mapping logic with explicit shape validation and parity tests.
2. Risk: char-level output may be whitespace-only causing flaky generation assertion.
   Mitigation: accept output IDs as generation evidence in integration probe.
3. Risk: tokenizer mismatch with chosen checkpoint.
   Mitigation: enforce tokenizer vocab-size check against HF config and fail fast with clear message.

## Implementation Checklist
- [x] Phase 1 Red: Add failing parity tests for HF nanoGPT config/mapping/fallback.
- [x] Phase 1 Green: Implement HF loader/mapping/fallback in `examples/web/nanogpt_example.rb`.
- [x] Phase 1 Refactor: Keep mapping helpers private and shape-validated.
- [x] Phase 2 Red: Update guardrail tests to target HF exporter path.
- [x] Phase 2 Green: Rewrite nanoGPT exporter to HF flow and regenerate assets.
- [x] Phase 2 Green: Update demo message and docs for `web:assets` source of truth.
- [x] Phase 3 Red: Capture integration assertion mismatch on empty rendered text.
- [x] Phase 3 Green: Update probe to validate text-or-token-id generation evidence.
- [x] Phase 3 Green: Run integration + wiring + guardrail + parity tests.
