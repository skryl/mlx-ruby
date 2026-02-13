# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase194ConvolutionTransposeLayersParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_conv_transpose1d_forward_with_bias
    conv = MLX::NN::ConvTranspose1d.new(1, 1, 2, bias: true)
    assert_equal [1, 2, 1], conv.weight.shape
    assert_equal [1], conv.bias.shape

    conv.weight = MLX::Core.array([[[1.0], [1.0]]], MLX::Core.float32)
    conv.bias = MLX::Core.array([0.25], MLX::Core.float32)
    x = MLX::Core.array([[[1.0], [2.0]]], MLX::Core.float32)

    out = conv.call(x)
    assert_nested_close [[[1.25], [3.25], [2.25]]], out.to_a
  end

  def test_conv_transpose2d_forward_with_bias
    conv = MLX::NN::ConvTranspose2d.new(1, 1, [2, 2], bias: true)
    assert_equal [1, 2, 2, 1], conv.weight.shape
    assert_equal [1], conv.bias.shape

    conv.weight = MLX::Core.array([[[[1.0], [1.0]], [[1.0], [1.0]]]], MLX::Core.float32)
    conv.bias = MLX::Core.array([0.5], MLX::Core.float32)
    x = MLX::Core.array([[[[1.0]]]], MLX::Core.float32)

    out = conv.call(x)
    assert_nested_close [[[[1.5], [1.5]], [[1.5], [1.5]]]], out.to_a
  end

  def test_conv_transpose3d_forward_without_bias
    conv = MLX::NN::ConvTranspose3d.new(1, 1, [2, 2, 2], bias: false)
    refute conv.state.key?("bias")
    assert_equal [1, 2, 2, 2, 1], conv.weight.shape

    conv.weight = MLX::Core.array(
      [[[[[1.0], [1.0]], [[1.0], [1.0]]], [[[1.0], [1.0]], [[1.0], [1.0]]]]],
      MLX::Core.float32
    )
    x = MLX::Core.array([[[[[1.0]]]]], MLX::Core.float32)
    out = conv.call(x)

    assert_nested_close [[[[[1.0], [1.0]], [[1.0], [1.0]]], [[[1.0], [1.0]], [[1.0], [1.0]]]]], out.to_a
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
