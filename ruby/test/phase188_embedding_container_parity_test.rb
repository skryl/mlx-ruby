# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase188EmbeddingContainerParityTest < Minitest::Test
  class ScaleModule < MLX::NN::Module
    def initialize(scale)
      super()
      @scale = scale
    end

    def call(x)
      MLX::Core.multiply(x, @scale)
    end
  end

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_embedding_lookup_and_as_linear
    emb = MLX::NN::Embedding.new(3, 2)
    assert_equal [3, 2], emb.weight.shape

    emb.weight = MLX::Core.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], MLX::Core.float32)

    idx = MLX::Core.array([[2, 0], [1, 2]], MLX::Core.int32)
    looked_up = emb.call(idx)
    assert_nested_close [[[30.0, 31.0], [10.0, 11.0]], [[20.0, 21.0], [30.0, 31.0]]], looked_up.to_a

    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    projected = emb.as_linear(x)
    assert_nested_close [[32.0, 62.0, 92.0], [74.0, 144.0, 214.0]], projected.to_a
  end

  def test_sequential_applies_layers_in_order
    seq = MLX::NN::Sequential.new(
      ScaleModule.new(2.0),
      ->(x) { MLX::Core.add(x, 1.0) },
      ScaleModule.new(3.0)
    )

    assert_equal 3, seq.layers.length

    x = MLX::Core.array([[1.0, 2.0]], MLX::Core.float32)
    y = seq.call(x)
    assert_nested_close [[9.0, 15.0]], y.to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal shape_signature(expected), shape_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |item| flatten(item) }
  end

  def shape_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |item| shape_signature(item) })]
  end
end
