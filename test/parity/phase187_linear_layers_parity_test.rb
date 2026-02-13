# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase187LinearLayersParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_identity_is_passthrough
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    layer = MLX::NN::Identity.new(:unused, foo: "bar")

    y = layer.call(x)
    assert_same x, y
    assert_nested_close [[1.0, 2.0], [3.0, 4.0]], y.to_a
  end

  def test_linear_initializes_shapes_and_applies_affine_transform
    layer = MLX::NN::Linear.new(3, 2, bias: true)
    assert_equal [2, 3], layer.weight.shape
    assert_equal [2], layer.bias.shape

    layer.weight = MLX::Core.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]], MLX::Core.float32)
    layer.bias = MLX::Core.array([0.5, -1.0], MLX::Core.float32)

    x = MLX::Core.array([[1.0, 2.0, 3.0], [0.0, -1.0, 1.0]], MLX::Core.float32)
    y = layer.call(x)

    assert_nested_close [[-1.5, 3.0], [-0.5, -2.0]], y.to_a
  end

  def test_linear_supports_no_bias
    layer = MLX::NN::Linear.new(2, 1, bias: false)
    refute layer.state.key?("bias")

    layer.weight = MLX::Core.array([[2.0, -3.0]], MLX::Core.float32)
    x = MLX::Core.array([[1.0, 5.0], [-2.0, 4.0]], MLX::Core.float32)

    y = layer.call(x)
    assert_nested_close [[-13.0], [-16.0]], y.to_a
  end

  def test_bilinear_initializes_shapes_and_applies_transform
    layer = MLX::NN::Bilinear.new(2, 3, 2, bias: true)
    assert_equal [2, 3, 2], layer.weight.shape
    assert_equal [2], layer.bias.shape

    layer.weight = MLX::Core.array(
      [
        [[1.0, 2.0], [0.0, 1.0], [1.0, -1.0]],
        [[-1.0, 0.0], [2.0, 1.0], [0.5, 0.5]]
      ],
      MLX::Core.float32
    )
    layer.bias = MLX::Core.array([0.5, -0.5], MLX::Core.float32)

    x1 = MLX::Core.array([[1.0, 2.0], [0.0, -1.0]], MLX::Core.float32)
    x2 = MLX::Core.array([[3.0, 4.0, 5.0], [1.0, 2.0, 3.0]], MLX::Core.float32)

    y = layer.call(x1, x2)
    assert_nested_close [[18.5, 20.0], [-0.5, -4.0]], y.to_a
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
