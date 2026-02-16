Runtime Primitives
==================

This page documents higher-level DSL helpers for transformer-style model code
and generation workflows.

Typed Config With ``ConfigSchema``
----------------------------------

Use ``MLX::DSL::ConfigSchema`` to declare typed config fields with defaults and
validation.

.. code-block:: ruby

   class LlamaConfig
     include MLX::DSL::ConfigSchema

     field :hidden_size, Integer, required: true
     field :num_heads, Integer, required: true
     field :num_kv_heads, Integer, default: ->(cfg) { cfg.num_heads }
     field :rope_theta, Float, default: 10_000.0
   end

   cfg = LlamaConfig.from_hash(hidden_size: 4096, num_heads: 32)
   cfg.to_h # => {"hidden_size"=>4096, ...}

Checkpoint Key Translation With ``weight_map``
----------------------------------------------

Use ``MLX::DSL.weight_map`` to translate key names and tensor layouts before
``load_weights``.

.. code-block:: ruby

   map = MLX::DSL.weight_map do
     strip_prefix "text_model."
     rename "attn." => "attention."
     split_qkv "attention.in_proj_weight",
               into: ["attention.q.weight", "attention.k.weight", "attention.v.weight"],
               axis: 0
     transpose_if rank: 4, order: [0, 2, 3, 1]
   end

   translated = map.apply(raw_weights)
   model.load_weights(translated.to_a, strict: false)

``KVCache`` State Management
----------------------------

Use ``MLX::DSL::KVCache`` for per-layer key/value cache lifecycle operations.

.. code-block:: ruby

   cache = MLX::DSL::KVCache.new(num_layers: 32)

   keys, values = cache.append(layer: 0, keys: k_step, values: v_step)
   offset = cache.offset(layer: 0)

   cache.truncate!(tokens: 128)  # keep most recent tokens
   cache.reset!(layer: 0)        # reset one layer
   cache.reset!                  # reset all layers

Mask And Position Helpers
-------------------------

Use ``Masks`` and ``Positions`` helpers to avoid repeating boilerplate for
causal decode paths.

.. code-block:: ruby

   mask = MLX::DSL::Masks.causal(
     length: seq_len,
     offset: cache.offset(layer: 0),
     dtype: hidden.dtype
   )

   pos_ids = MLX::DSL::Positions.ids_like(input_ids)
   rope_offset = MLX::DSL::Positions.offset_from_cache(cache, layer: 0)

Tensor Update Helpers
---------------------

Use tensor-native helpers instead of ``to_a`` mutation loops in Ruby.

.. code-block:: ruby

   merged = MLX::DSL::Tensor.scatter_rows(
     base: inputs_embeds,         # e.g. [B, T, D]
     row_indices: image_positions,
     values: image_features
   )

   point_emb = MLX::DSL::Tensor.where_labels(
     base: point_embedding,
     labels: point_labels,
     mapping: {
       -1 => not_a_point_embed.weight,
       0 => point_embed0.weight,
       1 => point_embed1.weight
     },
     mode: :add_or_replace
   )

Attention Builder
-----------------

``MLX::DSL::Attention`` wraps Q/K/V projection, optional grouped KV heads, RoPE,
and cache threading.

.. code-block:: ruby

   attn = MLX::DSL::Attention.new(
     dims: 4096,
     num_heads: 32,
     kv_heads: 8,
     cache: true,
     rope: { base: 10_000.0, traditional: true }
   )

   out, next_cache = attn.call(x, x, x, mask: mask, cache: layer_cache)

Builder shorthand:

.. code-block:: ruby

   layer :self_attn do
     attention dims: dims, num_heads: heads, kv_heads: kv_heads, cache: true
   end

``TransformerBlock`` Composition
--------------------------------

Use ``MLX::DSL::TransformerBlock`` for a pre-norm attention + feed-forward
residual block with configurable norm and FFN variant.

.. code-block:: ruby

   block = MLX::DSL::TransformerBlock.new(
     dims: 4096,
     num_heads: 32,
     kv_heads: 8,
     norm: :rms,
     ffn: { kind: :swiglu, hidden_dims: 14_336 },
     cache: true
   )

   hidden, next_cache = block.call(hidden, mask: mask, cache: cache_i)

Builder shorthand:

.. code-block:: ruby

   layer :block do
     transformer_block dims: dims, num_heads: heads, kv_heads: kv_heads, cache: true
   end

``run_stack`` For Layer Loops
-----------------------------

Use ``MLX::DSL.run_stack`` to execute a layer list while threading cache per
layer.

.. code-block:: ruby

   hidden, cache = MLX::DSL.run_stack(layers, hidden, mask: mask, cache: cache)

``cache`` can be:

- ``nil`` (returns only hidden state)
- an ``Array`` of per-layer cache entries
- ``MLX::DSL::KVCache``

Autoregressive ``Generate`` Helper
----------------------------------

Use ``MLX::DSL::Generate`` to run token loops for decoder-only or
encoder-decoder models.

.. code-block:: ruby

   generator = MLX::DSL::Generate.new(
     model: model,
     tokenizer: tokenizer,
     eos_id: tokenizer.eos_id,
     sampler: { strategy: :argmax },
     mode: :decoder_only
   )

   generator.each_token(prompt: "In the beginning", max_tokens: 128) do |token_id, chunk|
     print chunk if chunk
   end

Encoder-decoder mode:

.. code-block:: ruby

   generator = MLX::DSL::Generate.new(
     model: t5_model,
     tokenizer: tokenizer,
     eos_id: tokenizer.eos_id,
     mode: :encoder_decoder,
     decoder_start_id: tokenizer.decoder_start_id
   )

Implementation files:

- ``lib/mlx/dsl/config_schema.rb``
- ``lib/mlx/dsl/weight_map.rb``
- ``lib/mlx/dsl/kv_cache.rb``
- ``lib/mlx/dsl/masks.rb``
- ``lib/mlx/dsl/positions.rb``
- ``lib/mlx/dsl/tensor.rb``
- ``lib/mlx/dsl/attention.rb``
- ``lib/mlx/dsl/transformer_block.rb``
- ``lib/mlx/dsl/run_stack.rb``
- ``lib/mlx/dsl/generate.rb``
