# 2026_02_19 nanoGPT Web Demo RFP

## Status

In Progress (2026-02-19)

## Context

We need a Ruby nanoGPT example plus a browser demo backed by ONNX export.
The reference implementation is vendored in `examples/nanoGPT`.

This work should mirror the nanoGPT Shakespeare-char configuration:

1. dataset: Tiny Shakespeare (character-level),
2. model size: `n_layer=6`, `n_head=6`, `n_embd=384`, `block_size=256`,
3. training profile defaults from `train_shakespeare_char.py`.

## Goals

1. Add `examples/web/nanogpt_example.rb` defining a reusable nanoGPT model and
   dataset/tokenizer utilities for Shakespeare-char training.
2. Add web-side training/export scripts for this model.
3. Add a dedicated browser demo under `web/demo/nanogpt/`.
4. Integrate into existing `web:assets`, `web:start`, and docs flow.

## Non-goals

1. Reproducing full PyTorch nanoGPT training infrastructure.
2. Producing GPT-2 quality from Tiny Shakespeare.
3. Building a generic tokenizer framework beyond this demo path.

## Phased Plan (Red/Green)

### Phase 1: Ruby nanoGPT example

Red:
1. `examples/web/nanogpt_example.rb` does not exist and benchmark stub is empty.

Green:
1. Add `examples/web/nanogpt_example.rb` with:
   - model class (`NanoGptModel`)
   - Shakespeare-char dataset/tokenizer helpers
   - train loss and LR scheduler helpers
2. Keep defaults aligned with nanoGPT `train_shakespeare_char.py`.

Exit criteria:
1. File loads successfully with `ruby -c`.
2. Model forward accepts token IDs and returns logits.

### Phase 2: Web training/export integration

Red:
1. No web training/export flow exists for nanoGPT assets.

Green:
1. Add `web/assets/train_nanogpt_shakespeare.rb`.
2. Add `web/assets/export_nanogpt_assets.rb`.
3. Exporter supports:
   - random-init fallback
   - trained-checkpoint mode when artifacts exist.

Exit criteria:
1. Training script writes weights/tokenizer/config/summary artifacts.
2. Export script writes graph_ir/onnx/meta/presets artifacts.

### Phase 3: Browser demo + wiring

Red:
1. No `/demo/nanogpt/` route or index link exists.

Green:
1. Add `web/demo/nanogpt/index.html` and `web/demo/nanogpt/main.js`.
2. Update `web/index.html` with nanoGPT demo link.
3. Add `web:train[nanogpt]` and wire exporter into `web:assets` + `web:start`.
4. Update `web/README.md`.

Exit criteria:
1. `rake web:start` serves demo index and `/demo/nanogpt/`.
2. Demo loads ONNX model + runs generation loop with provider fallback.

## Acceptance Criteria

1. `examples/web/nanogpt_example.rb` exists and reflects nanoGPT Shakespeare-char
   size/data defaults.
2. `bundle exec rake web:train` produces checkpoint artifacts.
3. `bundle exec rake web:assets` produces nanoGPT ONNX assets.
4. `bundle exec rake web:start` serves the new nanoGPT demo.
5. README documents training and launch commands.

## Risks & Mitigations

1. Risk: model export too large/slow for browser startup.
   Mitigation: keep demo generation settings conservative and rely on local
   serving; allow retraining with fewer steps.
2. Risk: tokenizer mismatch across train/export/demo.
   Mitigation: persist tokenizer JSON and consume it in exporter and JS.
3. Risk: long training time for default 5k steps.
   Mitigation: expose env overrides for fast local smoke loops.

## Implementation Checklist

- [ ] Phase 1 red check captured.
- [ ] Phase 1 green implementation complete.
- [ ] Phase 1 exit criteria met.
- [ ] Phase 2 red check captured.
- [ ] Phase 2 green implementation complete.
- [ ] Phase 2 exit criteria met.
- [ ] Phase 3 red check captured.
- [ ] Phase 3 green implementation complete.
- [ ] Phase 3 exit criteria met.
