# frozen_string_literal: true

require_relative "test_helper"

class Phase48ConvolutionAdvancedTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_conv_transpose1d
    input = MLX::Core.array([[[1.0], [2.0]]], MLX::Core.float32)
    weight = MLX::Core.array([[[1.0], [1.0]]], MLX::Core.float32)

    out = MLX::Core.conv_transpose1d(input, weight)
    assert_equal [1, 3, 1], out.shape
    assert_nested_close [[[1.0], [3.0], [2.0]]], out.to_a
  end

  def test_conv_transpose2d_and_conv_transpose3d
    input2 = MLX::Core.array([[[[1.0]]]], MLX::Core.float32)
    weight2 = MLX::Core.array([[[[1.0], [1.0]], [[1.0], [1.0]]]], MLX::Core.float32)
    out2 = MLX::Core.conv_transpose2d(input2, weight2)
    assert_equal [1, 2, 2, 1], out2.shape
    assert_nested_close [[[[1.0], [1.0]], [[1.0], [1.0]]]], out2.to_a

    input3 = MLX::Core.array([[[[[1.0]]]]], MLX::Core.float32)
    weight3 = MLX::Core.array([[[[[1.0], [1.0]], [[1.0], [1.0]]], [[[1.0], [1.0]], [[1.0], [1.0]]]]], MLX::Core.float32)
    out3 = MLX::Core.conv_transpose3d(input3, weight3)
    assert_equal [1, 2, 2, 2, 1], out3.shape
    assert_nested_close [[[[[1.0], [1.0]], [[1.0], [1.0]]], [[[1.0], [1.0]], [[1.0], [1.0]]]]], out3.to_a
  end

  def test_conv_general
    input = MLX::Core.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], MLX::Core.float32)
    weight = MLX::Core.array([[[[1.0], [1.0]], [[1.0], [1.0]]]], MLX::Core.float32)

    out = MLX::Core.conv_general(input, weight, 1, 0, 1, 1, 1, false)
    assert_equal [1, 1, 1, 1], out.shape
    assert_nested_close [[[[10.0]]]], out.to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-4)
    assert_equal structure_signature(expected), structure_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |x| flatten(x) }
  end

  def structure_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |v| structure_signature(v) })]
  end
end
