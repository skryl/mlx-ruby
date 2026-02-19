# GPT-2 Example + Web Demo RFP

## Status
Completed (2026-02-19)

## Context
We need a first-class MLX-Ruby GPT-2 example that can load Hugging Face GPT-2
weights, and we need to expose GPT-2 as an additional web demo entry.

## Goals
1. Add `examples/gpt2_example.rb` with a GPT-2 model architecture in Ruby.
2. Support loading Hugging Face GPT-2 state dict tensors (`model.safetensors`)
   into the Ruby model.
3. Add GPT-2 as an additional web demo route.
4. Add tests that validate HF weight mapping behavior.

## Non-goals
1. Training GPT-2 in this phase.
2. Supporting every GPT-2 family variant beyond base config fields.
3. Replacing existing language-model demos.

## Phased Plan

### Phase 1: Red test for HF mapping
- Red:
  - Add failing parity test for GPT-2 HF state mapping and forward shape.
- Green:
  - Implement GPT-2 modules and HF mapping to pass tests.
- Exit criteria:
  - New parity test passes locally.

### Phase 2: Web demo integration
- Red:
  - Add/verify route and docs expectations for a new `/demo/gpt2/` page.
- Green:
  - Add `web/demo/gpt2/` and index/docs links.
- Exit criteria:
  - `web:start` serves `/demo/gpt2/` successfully.

## Acceptance Criteria
1. `examples/gpt2_example.rb` defines a GPT-2 model with HF loader.
2. HF Conv1D-style weights are mapped correctly to MLX `Linear` parameters.
3. `test/parity/phase328_gpt2_example_weight_mapping_test.rb` is green.
4. Web index links include `/demo/gpt2/` as an additional example.

## Risks and Mitigations
1. Risk: Conv1D/Linear transpose mistakes.
   Mitigation: explicit red/green assertions for mapped tensors.
2. Risk: Web runtime dtype/input mismatch.
   Mitigation: rely on captured model metadata and validated feed construction.

## Implementation Checklist
- [x] Add parity test for GPT-2 HF mapping.
- [x] Implement `examples/gpt2_example.rb`.
- [x] Add HF loader + mapping helpers.
- [x] Add `web/demo/gpt2/` route.
- [x] Update index/docs links.
- [x] Run targeted tests and syntax checks.
