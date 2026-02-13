# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase198TransformerLayersParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_multihead_attention_validation_and_shape
    assert_raises(ArgumentError) { MLX::NN::MultiHeadAttention.new(5, 2) }

    mha = MLX::NN::MultiHeadAttention.new(4, 2, bias: false)
    q = MLX::Core.zeros([2, 3, 4], MLX::Core.float32)
    out = mha.call(q, q, q, nil)
    assert_equal [2, 3, 4], out.shape

    mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(3)
    assert_equal [3, 3], mask.shape
    assert_in_delta 0.0, mask.to_a[1][0], 1e-6
    assert_operator mask.to_a[0][1], :<, -1e10
  end

  def test_transformer_encoder_decoder_layer_shapes
    enc = MLX::NN::TransformerEncoderLayer.new(4, 2, mlp_dims: 8, dropout: 0.0, norm_first: true)
    dec = MLX::NN::TransformerDecoderLayer.new(4, 2, mlp_dims: 8, dropout: 0.0, norm_first: true)

    x = MLX::Core.zeros([2, 3, 4], MLX::Core.float32)
    memory = MLX::Core.zeros([2, 3, 4], MLX::Core.float32)
    mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(3)

    out_enc = enc.call(x, mask)
    out_dec = dec.call(x, memory, mask, nil)

    assert_equal [2, 3, 4], out_enc.shape
    assert_equal [2, 3, 4], out_dec.shape
  end

  def test_transformer_end_to_end_shape
    transformer = MLX::NN::Transformer.new(
      dims: 4,
      num_heads: 2,
      num_encoder_layers: 1,
      num_decoder_layers: 1,
      mlp_dims: 8,
      dropout: 0.0
    )

    src = MLX::Core.zeros([2, 3, 4], MLX::Core.float32)
    tgt = MLX::Core.zeros([2, 2, 4], MLX::Core.float32)
    src_mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(3)
    tgt_mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(2)

    out = transformer.call(src, tgt, src_mask, tgt_mask, nil)
    assert_equal [2, 2, 4], out.shape
  end
end
