# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase192QuantizedScaffoldingParityTest < Minitest::Test
  class QuantizeToyModel < MLX::NN::Module
    def initialize
      super()
      self.linear = MLX::NN::Linear.new(64, 2)
      self.embed = MLX::NN::Embedding.new(4, 64)
      self.dropout = MLX::NN::Dropout.new(0.0)
    end
  end

  def setup
    TestSupport.build_native_extension!
    MLX::Core.random_seed(42)
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_to_quantized_routes_for_linear_and_embedding
    linear = MLX::NN::Linear.new(64, 3)
    q_linear = linear.to_quantized
    assert_instance_of MLX::NN::QuantizedLinear, q_linear

    linear_no_bias = MLX::NN::Linear.new(64, 3, bias: false)
    q_qlinear = linear_no_bias.to_quantized(mode: "nvfp4", quantize_input: true)
    assert_instance_of MLX::NN::QQLinear, q_qlinear

    assert_raises(ArgumentError) do
      linear.to_quantized(mode: "affine", quantize_input: true)
    end

    embedding = MLX::NN::Embedding.new(5, 64)
    q_embedding = embedding.to_quantized
    assert_instance_of MLX::NN::QuantizedEmbedding, q_embedding

    assert_raises(ArgumentError) do
      embedding.to_quantized(quantize_input: true)
    end
  end

  def test_from_linear_and_from_embedding_scaffolding
    linear = MLX::NN::Linear.new(64, 2, bias: true)
    linear.weight = MLX::Core.reshape(MLX::Core.arange(0, 128, 1, MLX::Core.float32), [2, 64])
    linear.bias = MLX::Core.array([0.25, -0.75], MLX::Core.float32)
    q_linear = MLX::NN::QuantizedLinear.from_linear(linear, mode: "affine")

    x = MLX::Core.ones([1, 64], MLX::Core.float32)
    y = q_linear.call(x)
    assert_equal [1, 2], y.shape
    assert_nested_close [0.25, -0.75], q_linear.bias.to_a

    embedding = MLX::NN::Embedding.new(3, 64)
    q_embedding = MLX::NN::QuantizedEmbedding.from_embedding(embedding, mode: "affine")

    idx = MLX::Core.array([[2, 0]], MLX::Core.int32)
    looked_up = q_embedding.call(idx)
    assert_equal [1, 2, 64], looked_up.shape

    projected = q_embedding.as_linear(MLX::Core.ones([1, 64], MLX::Core.float32))
    assert_equal [1, 3], projected.shape
  end

  def test_quantize_model_updates_leaf_modules
    model = QuantizeToyModel.new
    MLX::NN.quantize(model, mode: "affine")

    assert_instance_of MLX::NN::QuantizedLinear, model.linear
    assert_instance_of MLX::NN::QuantizedEmbedding, model.embed
    assert_instance_of MLX::NN::Dropout, model.dropout
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-3)
    expected.flatten.zip(actual.flatten).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end
end
