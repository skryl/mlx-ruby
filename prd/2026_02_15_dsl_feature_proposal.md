# Ruby DSL Feature Proposal (Model Ergonomics)

Date: 2026-02-15

## Scope

This proposal is based on reading all model-definition files in `../../mlx-ruby-examples/codex-examples` (38 files, ~8.5k LOC).

Goal: reduce repetition, boilerplate, and abstraction leakage in model code while keeping compatibility with existing `MLX::NN::Module`, `MLX::Core`, and current DSL surfaces.

Note: API names below are proposal-level and can be adjusted during implementation.

## Summary Of Proposed Features

1. `transformer_block` macro
2. `attention` module builder with cache/rope/GQA support
3. `run_stack` helper for repeated layer + cache threading
4. unified `KVCache` DSL/state API
5. mask/position helper APIs
6. tensor-native scatter/update helpers (replace `to_a` patching)
7. `weight_map` DSL for checkpoint key translation
8. `config_schema` macro for typed config objects
9. `generate` runtime helper for autoregressive loops

---

## 1) `transformer_block` Macro

### Problem

The same transformer block skeleton is reimplemented many times:

- pre-norm attention residual
- pre-norm FFN residual
- optional cross-attention
- optional gated FFN variants

Examples:

- `../../mlx-ruby-examples/codex-examples/llms/llama/llama.rb:163`
- `../../mlx-ruby-examples/codex-examples/llms/mistral/mistral.rb:149`
- `../../mlx-ruby-examples/codex-examples/llava/language.rb:180`
- `../../mlx-ruby-examples/codex-examples/t5/model.rb:317`

### Proposed API

```ruby
class DecoderBlock < MLX::DSL::Model
  option :dims
  option :num_heads
  option :kv_heads, default: -> { num_heads }
  option :ffn_dims, default: -> { dims * 4 }
  option :norm_eps, default: 1e-6

  layer :block do
    transformer_block(
      dims: dims,
      num_heads: num_heads,
      kv_heads: kv_heads,
      norm: :rms,
      norm_eps: norm_eps,
      ffn: { kind: :swiglu, hidden_dims: ffn_dims },
      cache: true,
      rope: { base: 10_000.0, traditional: false }
    )
  end

  def call(x, mask: nil, cache: nil)
    block.call(x, mask: mask, cache: cache)
  end
end
```

---

## 2) `attention` Builder With Cache/RoPE/GQA

### Problem

Attention internals are repeatedly hand-written:

- q/k/v projections
- reshape/transpose and merge back
- kv cache append
- RoPE offset logic
- kv head replication for GQA/MQA

Examples:

- `../../mlx-ruby-examples/codex-examples/llms/llama/llama.rb:93`
- `../../mlx-ruby-examples/codex-examples/llms/mixtral/mixtral.rb:96`
- `../../mlx-ruby-examples/codex-examples/llava/language.rb:124`
- `../../mlx-ruby-examples/codex-examples/t5/model.rb:206`

### Proposed API

```ruby
class SelfAttn < MLX::DSL::Model
  option :dims
  option :num_heads
  option :kv_heads, default: -> { num_heads }

  layer :attn do
    attention(
      dims: dims,
      num_heads: num_heads,
      kv_heads: kv_heads,
      qkv_bias: false,
      backend: :sdpa,
      rope: { base: 10_000.0, traditional: true },
      cache: true
    )
  end

  def call(x, mask: nil, cache: nil)
    attn.call(x, x, x, mask: mask, cache: cache)
  end
end
```

---

## 3) `run_stack` Helper For Layer Loops

### Problem

Many models duplicate:

- `self.layers = Array.new(depth) { ... }`
- forward loops over `layers`
- cache arrays initialized and threaded via `each_with_index`

Examples:

- `../../mlx-ruby-examples/codex-examples/llms/mistral/mistral.rb:178`
- `../../mlx-ruby-examples/codex-examples/llms/mistral/mistral.rb:193`
- `../../mlx-ruby-examples/codex-examples/t5/model.rb:347`
- `../../mlx-ruby-examples/codex-examples/t5/model.rb:374`

### Proposed API

```ruby
class DecoderStack < MLX::DSL::Model
  option :depth
  option :dims
  option :num_heads

  layer :blocks do
    stack(depth) do
      transformer_block(dims: dims, num_heads: num_heads, norm: :rms, cache: true)
    end
  end

  def call(x, mask: nil, cache: nil)
    run_stack(blocks.layers, x, mask: mask, cache: cache)
    # => [hidden, next_cache]
  end
end
```

---

## 4) Unified `KVCache` DSL API

### Problem

Cache lifecycle logic (offset, append, truncate, reset) is inconsistent and repeated.

Examples:

- `../../mlx-ruby-examples/codex-examples/t5/model.rb:354`
- `../../mlx-ruby-examples/codex-examples/t5/model.rb:430`
- `../../mlx-ruby-examples/codex-examples/llms/speculative_decoding/model.rb:295`
- `../../mlx-ruby-examples/codex-examples/musicgen/musicgen.rb:152`

### Proposed API

```ruby
cache = MLX::DSL::KVCache.new(num_layers: num_layers)

hidden, cache = run_stack(layers, hidden, mask: mask, cache: cache)
offset = cache.offset(layer: 0)

cache.truncate!(tokens: 128) # keep most recent 128 cached steps
cache.reset!
```

Inside attention:

```ruby
keys, values = cache.append(layer: layer_idx, keys: keys, values: values)
```

---

## 5) Mask And Position Helpers

### Problem

Causal masks, position IDs, and offset handling are frequently duplicated.

Examples:

- `../../mlx-ruby-examples/codex-examples/t5/model.rb:156`
- `../../mlx-ruby-examples/codex-examples/llms/speculative_decoding/model.rb:96`
- `../../mlx-ruby-examples/codex-examples/whisper/whisper.rb:94`
- `../../mlx-ruby-examples/codex-examples/bert/model.rb:95`

### Proposed API

```ruby
mask = MLX::DSL::Masks.causal(
  length: seq_len,
  offset: cache.offset(layer: 0),
  dtype: hidden.dtype
)

pos_ids = MLX::DSL::Positions.ids_like(input_ids)  # [batch, seq]
rope_offset = MLX::DSL::Positions.offset_from_cache(cache, layer: 0)
```

---

## 6) Tensor-Native Scatter/Update Helpers

### Problem

Some model code converts tensors to Ruby arrays (`to_a`) and mutates in Ruby loops.
This is verbose, slow, and leaks low-level representation details.

Examples:

- `../../mlx-ruby-examples/codex-examples/llava/llava.rb:133`
- `../../mlx-ruby-examples/codex-examples/llava/llava.rb:144`
- `../../mlx-ruby-examples/codex-examples/segment_anything/prompt_encoder.rb:142`
- `../../mlx-ruby-examples/codex-examples/segment_anything/prompt_encoder.rb:169`

### Proposed API

```ruby
merged = MLX::DSL::Tensor.scatter_rows(
  base: inputs_embeds,              # [B, T, D]
  row_indices: image_token_positions, # [P]
  values: image_features            # [P, D]
)
```

Label-based update:

```ruby
emb = MLX::DSL::Tensor.where_labels(
  base: point_embedding,
  labels: point_labels,
  mapping: {
    -1 => not_a_point_embed.weight,
    0 => point_embed[0].weight,
    1 => point_embed[1].weight
  },
  mode: :add_or_replace
)
```

---

## 7) `weight_map` DSL For Checkpoint Translation

### Problem

Many loaders implement ad hoc key-renaming/transpose/splitting logic for imported checkpoints.

Examples:

- `../../mlx-ruby-examples/codex-examples/llms/gguf_llm/models.rb:270`
- `../../mlx-ruby-examples/codex-examples/flux/flux/clip.rb:100`
- `../../mlx-ruby-examples/codex-examples/flux/flux/model.rb:94`
- `../../mlx-ruby-examples/codex-examples/musicgen/musicgen.rb:391`

### Proposed API

```ruby
map = MLX::DSL.weight_map do
  strip_prefix "text_model."
  strip_prefix "encoder."

  rename "self_attn.q_proj." => "attention.query_proj."
  rename "self_attn.k_proj." => "attention.key_proj."
  rename "self_attn.v_proj." => "attention.value_proj."
  rename "mlp.fc1" => "linear1"
  rename "mlp.fc2" => "linear2"

  split_qkv "attn.in_proj_weight",
            into: ["attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight"],
            axis: 0

  transpose_if rank: 4, order: [0, 2, 3, 1]
end

model.load_weights(map.apply(raw_weights).to_a, strict: false)
```

---

## 8) `config_schema` Macro

### Problem

Config objects are repeated with near-identical mechanics:

- initializer args/defaults
- `from_hash` / `from_dict`
- key coercion and validation
- `to_h`

Examples:

- `../../mlx-ruby-examples/codex-examples/llava/language.rb:7`
- `../../mlx-ruby-examples/codex-examples/musicgen/musicgen.rb:16`
- `../../mlx-ruby-examples/codex-examples/encodec/encodec.rb:54`
- `../../mlx-ruby-examples/codex-examples/lora/models.rb:10`

### Proposed API

```ruby
class LlamaConfig
  include MLX::DSL::ConfigSchema

  field :hidden_size, Integer, required: true
  field :num_hidden_layers, Integer, required: true
  field :num_attention_heads, Integer, required: true
  field :num_key_value_heads, Integer, default: ->(cfg) { cfg.num_attention_heads }
  field :rope_theta, Float, default: 10_000.0
  field :rope_scaling, Hash, default: nil do |value|
    next if value.nil?
    raise ArgumentError, "rope_scaling requires factor/type" unless value.key?("factor") && value.key?("type")
  end
end

cfg = LlamaConfig.from_hash(raw_config_hash)
cfg.to_h
```

---

## 9) `generate` Runtime Helper

### Problem

Autoregressive generation loops are reimplemented across examples:

- prompt prefill
- iterative decode with cache
- temperature/top-k sampling
- EOS handling
- streamed output buffering

Examples:

- `../../mlx-ruby-examples/codex-examples/llms/llama/llama.rb:200`
- `../../mlx-ruby-examples/codex-examples/llms/mistral/mistral.rb:251`
- `../../mlx-ruby-examples/codex-examples/llms/mixtral/mixtral.rb:289`
- `../../mlx-ruby-examples/codex-examples/t5/model.rb:462`
- `../../mlx-ruby-examples/codex-examples/musicgen/musicgen.rb:351`

### Proposed API

```ruby
generator = MLX::DSL::Generate.new(
  model: model,
  tokenizer: tokenizer,
  eos_id: tokenizer.eos_id,
  sampler: { strategy: :top_k, k: 40, temperature: 0.8 }
)

generator.each_token(prompt: "In the beginning", max_tokens: 256) do |token_id, text_chunk|
  print text_chunk if text_chunk
end
```

T5-like encoder/decoder support:

```ruby
generator = MLX::DSL::Generate.new(
  model: t5_model,
  mode: :encoder_decoder,
  decoder_start_id: tokenizer.decoder_start_id
)
```

---

## Acceptance Criteria

The proposal is considered successfully implemented when all criteria below are met.

### API Stability And Ergonomics

- New DSL APIs are additive and do not break existing `MLX::DSL::Model`, `ModelMixin`, or `Trainer` usage.
- Each proposed feature has docs with at least one minimal runnable snippet.
- At least one existing example per feature area is migrated or wrapped to demonstrate reduced boilerplate.

### Correctness

- New abstractions produce numerically equivalent outputs to previous hand-written implementations for representative models.
- Cache-aware paths (`KVCache`, decoder stacks, generation loops) preserve token-by-token parity against current behavior.
- Mask/position helper outputs match existing model-specific implementations.

### Performance And Memory

- Replacing tensor logic with DSL helpers must not introduce Ruby `to_a` round-trips in hot paths.
- New helpers avoid additional per-step object churn in generation and decoder loops.
- Feature wrappers keep existing attention backends (`scaled_dot_product_attention` / equivalent) unless explicitly configured otherwise.

### Test Coverage

- Unit tests exist for each new API surface.
- Integration tests cover at least:
  - one decoder-only model path (`llama`/`mistral`-style),
  - one encoder-decoder path (`t5`-style),
  - one multimodal merge/scatter path (`llava`/`sam`-style),
  - one checkpoint key-translation path (`weight_map`).
- Backward-compatibility tests validate unchanged behavior of existing DSL APIs.

### Adoption Targets

- Boilerplate reduction is measurable in migrated examples (fewer lines in core model classes; fewer repeated attention/cache utilities).
- Reused helper count increases (shared stack/cache/mask helpers replacing per-example copies).

---

## Implementation Checklist

### 0. Foundations

- [ ] Add `rfp`/docs cross-links for new DSL feature docs.
- [ ] Add shared naming conventions for new DSL APIs (finalize module/class/method names).
- [ ] Add deprecation policy notes for any superseded helper patterns (if applicable).

### 1. `attention` Builder + `KVCache` + Mask/Position Helpers

- [ ] Implement `MLX::DSL::Attention` builder/module wrapper with qkv projection + head packing utilities.
- [ ] Add options for `num_heads`, `kv_heads`, optional RoPE, cache append, and backend selection.
- [ ] Implement `MLX::DSL::KVCache` with `offset`, `append`, `truncate!`, `reset!`, and per-layer state access.
- [ ] Implement `MLX::DSL::Masks.causal(...)` and position helper APIs.
- [ ] Add unit tests for:
  - [ ] head shape transforms,
  - [ ] cache growth and truncate semantics,
  - [ ] mask equivalence against prior implementations,
  - [ ] RoPE offset behavior.

### 2. `run_stack` + `transformer_block`

- [ ] Implement `run_stack` for pure stack and cache-threaded stack execution.
- [ ] Implement `transformer_block(...)` composition macro (norm choice, FFN kind, optional cross-attn).
- [ ] Add tests for:
  - [ ] decoder-only block behavior,
  - [ ] cross-attention block behavior,
  - [ ] cache threading across N layers.
- [ ] Migrate at least one decoder model to validate API fit.

### 3. Tensor Scatter/Update Helpers

- [ ] Implement tensor-native row/token scatter helper(s) replacing `to_a` mutation workflows.
- [ ] Implement label-conditioned update helper for prompt-embedding style use cases.
- [ ] Add tests for shape safety, index validation, duplicate index policy, dtype/device preservation.
- [ ] Migrate one `llava` or `segment_anything` hotspot path to use helper(s).

### 4. `weight_map` DSL

- [ ] Implement `MLX::DSL.weight_map` builder with:
  - [ ] prefix stripping,
  - [ ] rename rules,
  - [ ] regex rule support,
  - [ ] qkv split rule,
  - [ ] rank-based transpose rule.
- [ ] Add deterministic rule-order behavior and conflict handling.
- [ ] Add tests with representative mappings from `gguf`, `flux`, and `musicgen` patterns.

### 5. `config_schema` Macro

- [ ] Implement `ConfigSchema` with typed `field` declarations, required/default handling, and validation hooks.
- [ ] Support `from_hash` + `to_h` generation with symbol/string key normalization.
- [ ] Add tests for type coercion, missing fields, custom validators, default lambdas.
- [ ] Migrate one or two config classes to validate ergonomics.

### 6. `generate` Runtime Helper

- [ ] Implement unified `Generate` helper for decoder-only and encoder-decoder modes.
- [ ] Add sampler support (`argmax`, `temperature`, `top_k`) and EOS stop behavior.
- [ ] Add streaming token callback/iterator API.
- [ ] Add tests for cache reuse, EOS termination, and deterministic output in `temp=0` mode.

### 7. Integration, Migration, And Hardening

- [ ] Add migration examples showing before/after reductions in model boilerplate.
- [ ] Add integration tests across at least three model families.
- [ ] Benchmark critical decode loop paths for regressions.
- [ ] Finalize docs and API reference updates.
- [ ] Mark checklist complete only after CI passes with new tests enabled.

---

## Recommended Delivery Order

1. `attention` builder + `KVCache` API + mask/position helpers (largest repetition win)
2. `run_stack` + `transformer_block` macro
3. tensor scatter/update helpers
4. `weight_map` DSL and `config_schema`
5. `generate` runtime helper

## Compatibility Notes

- New DSL APIs should compile to existing `MLX::NN` modules and `MLX::Core` ops.
- No breaking changes to current `MLX::DSL::Model`, `ModelMixin`, `Trainer`.
- Features should be adoptable incrementally in examples.
