# frozen_string_literal: true

require_relative "../test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class DslRfpFeaturesUnitTest < Minitest::Test
  def setup
    begin
      TestSupport.build_native_extension!
    rescue StandardError => e
      skip "Native extension build unavailable in this environment: #{e.message.lines.first&.strip || e.message}"
    end
    skip "MLX native extension unavailable" unless MLX.native_available?
  end

  class ConfigSchemaExample
    include MLX::DSL::ConfigSchema

    field :hidden_size, Integer, required: true
    field :num_heads, Integer, required: true
    field :num_kv_heads, Integer, default: lambda { |cfg|
      cfg.num_heads
    }
    field :rope_theta, Float, default: 10_000.0
  end

  class CacheInc < MLX::NN::Module
    def call(x, cache: nil, **_kwargs)
      next_cache = cache.nil? ? 1 : cache + 1
      [x + 1, next_cache]
    end
  end

  class TinyDecodeModel < MLX::NN::Module
    def call(input_ids, cache: nil)
      step = cache.to_i
      token = step % 3
      seq_len = input_ids.shape[1]
      logits = Array.new(1) { Array.new(seq_len) { Array.new(5, 0.0) } }
      logits[0][seq_len - 1][token] = 10.0
      [MLX::Core.array(logits, MLX::Core.float32), step + 1]
    end
  end

  class TinyTokenizer
    def encode(text)
      text.to_s.bytes
    end

    def decode(tokens)
      tokens.map { |t| t.to_i.chr(Encoding::UTF_8) }.join
    end
  end

  def test_config_schema_from_hash_defaults_and_to_h
    cfg = ConfigSchemaExample.from_hash(hidden_size: 64, num_heads: 8)
    assert_equal 64, cfg.hidden_size
    assert_equal 8, cfg.num_heads
    assert_equal 8, cfg.num_kv_heads
    assert_equal 10_000.0, cfg.rope_theta
    assert_equal(
      {
        "hidden_size" => 64,
        "num_heads" => 8,
        "num_kv_heads" => 8,
        "rope_theta" => 10_000.0
      },
      cfg.to_h
    )
  end

  def test_config_schema_type_validation
    error = assert_raises(TypeError) do
      ConfigSchemaExample.from_hash(hidden_size: "64", num_heads: 8)
    end
    assert_match(/hidden_size/i, error.message)
  end

  def test_weight_map_supports_strip_rename_split_and_transpose
    raw = {
      "text_model.attn.in_proj_weight" => MLX::Core.reshape(
        MLX::Core.arange(0, 24, 1, MLX::Core.float32),
        [6, 4]
      ),
      "text_model.conv.weight" => MLX::Core.reshape(
        MLX::Core.arange(0, 24, 1, MLX::Core.float32),
        [1, 2, 3, 4]
      )
    }

    map = MLX::DSL.weight_map do
      strip_prefix "text_model."
      rename "attn." => "attention."
      split_qkv "attention.in_proj_weight",
                into: ["attention.q.weight", "attention.k.weight", "attention.v.weight"],
                axis: 0
      transpose_if rank: 4, order: [0, 2, 3, 1]
    end

    out = map.apply(raw)
    assert out.key?("attention.q.weight")
    assert out.key?("attention.k.weight")
    assert out.key?("attention.v.weight")
    assert out.key?("conv.weight")
    assert_equal [1, 3, 4, 2], out.fetch("conv.weight").shape
  end

  def test_kv_cache_append_truncate_and_reset
    cache = MLX::DSL::KVCache.new(num_layers: 2)
    k1 = MLX::Core.zeros([1, 2, 3, 4], MLX::Core.float32)
    v1 = MLX::Core.zeros([1, 2, 3, 4], MLX::Core.float32)
    k2 = MLX::Core.ones([1, 2, 2, 4], MLX::Core.float32)
    v2 = MLX::Core.ones([1, 2, 2, 4], MLX::Core.float32)

    keys, _values = cache.append(layer: 0, keys: k1, values: v1)
    assert_equal [1, 2, 3, 4], keys.shape
    assert_equal 3, cache.offset(layer: 0)

    keys, _values = cache.append(layer: 0, keys: k2, values: v2)
    assert_equal [1, 2, 5, 4], keys.shape
    assert_equal 5, cache.offset(layer: 0)

    cache.truncate!(tokens: 2, layer: 0)
    truncated = cache.layer(0).first
    assert_equal [1, 2, 2, 4], truncated.shape
    assert_equal 2, cache.offset(layer: 0)

    cache.reset!(layer: 0)
    assert_nil cache.layer(0)
    assert_equal 0, cache.offset(layer: 0)
  end

  def test_masks_and_positions_helpers
    ids = MLX::Core.zeros([2, 4], MLX::Core.int32)
    pos = MLX::DSL::Positions.ids_like(ids)
    assert_equal [[0, 1, 2, 3], [0, 1, 2, 3]], pos.to_a

    mask = MLX::DSL::Masks.causal(length: 2, offset: 1, dtype: MLX::Core.float32)
    assert_equal [2, 3], mask.shape
    values = mask.to_a
    assert_equal 0.0, values[0][0]
    assert values[0][2] < -1.0
    assert_equal 0.0, values[1][2]
  end

  def test_tensor_scatter_rows_and_where_labels
    base = MLX::Core.zeros([4, 2], MLX::Core.float32)
    indices = MLX::Core.array([1, 3], MLX::Core.int32)
    values = MLX::Core.array([[5.0, 6.0], [7.0, 8.0]], MLX::Core.float32)
    scattered = MLX::DSL::Tensor.scatter_rows(base: base, row_indices: indices, values: values)
    assert_nested_close [[0.0, 0.0], [5.0, 6.0], [0.0, 0.0], [7.0, 8.0]], scattered.to_a

    embeddings = MLX::Core.zeros([1, 3, 2], MLX::Core.float32)
    labels = MLX::Core.array([[-1, 0, 1]], MLX::Core.int32)
    mapped = MLX::DSL::Tensor.where_labels(
      base: embeddings,
      labels: labels,
      mapping: {
        -1 => MLX::Core.array([9.0, 9.0], MLX::Core.float32),
        0 => MLX::Core.array([1.0, 1.0], MLX::Core.float32),
        1 => MLX::Core.array([2.0, 2.0], MLX::Core.float32)
      },
      mode: :add_or_replace
    )
    assert_nested_close [[[9.0, 9.0], [1.0, 1.0], [2.0, 2.0]]], mapped.to_a
  end

  def test_attention_builder_supports_cache_and_kv_heads
    attn = MLX::DSL::Attention.new(dims: 4, num_heads: 2, kv_heads: 1, cache: true)
    x = MLX::Core.ones([1, 2, 4], MLX::Core.float32)

    out, cache = attn.call(x, x, x)
    assert_equal [1, 2, 4], out.shape
    assert_equal 2, cache.first.shape[2]

    x2 = MLX::Core.ones([1, 1, 4], MLX::Core.float32)
    _out2, cache2 = attn.call(x2, x2, x2, cache: cache)
    assert_equal 3, cache2.first.shape[2]
  end

  def test_transformer_block_and_run_stack
    block = MLX::DSL::TransformerBlock.new(dims: 4, num_heads: 2, cache: true)
    x = MLX::Core.ones([1, 2, 4], MLX::Core.float32)
    y, cache = block.call(x)
    assert_equal [1, 2, 4], y.shape
    refute_nil cache

    layers = [CacheInc.new, CacheInc.new]
    out, cache_arr = MLX::DSL.run_stack(layers, 0, cache: [])
    assert_equal 2, out
    assert_equal [1, 1], cache_arr
  end

  def test_generate_helper_runs_decoder_only_loop
    model = TinyDecodeModel.new
    tokenizer = TinyTokenizer.new
    generator = MLX::DSL::Generate.new(
      model: model,
      tokenizer: tokenizer,
      eos_id: 1,
      sampler: { strategy: :argmax },
      mode: :decoder_only
    )

    seen = []
    generator.each_token(prompt: "abc", max_tokens: 4) do |token_id, _chunk|
      seen << token_id
    end

    assert_equal [0, 1], seen
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    if expected.is_a?(Array)
      assert_instance_of Array, actual
      assert_equal expected.length, actual.length
      expected.each_with_index do |item, index|
        assert_nested_close(item, actual[index], atol)
      end
    else
      assert_in_delta expected.to_f, actual.to_f, atol
    end
  end
end

$LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
