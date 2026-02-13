# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase195PoolingLayersParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_pool1d_max_and_avg
    x = MLX::Core.array([[[1.0], [3.0], [2.0], [4.0]]], MLX::Core.float32)

    max_pool = MLX::NN::MaxPool1d.new(2, stride: 2)
    avg_pool = MLX::NN::AvgPool1d.new(2, stride: 1)

    assert_nested_close [[[3.0], [4.0]]], max_pool.call(x).to_a
    assert_nested_close [[[2.0], [2.5], [3.0]]], avg_pool.call(x).to_a
  end

  def test_pool2d_max_and_avg
    x = MLX::Core.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], MLX::Core.float32)

    max_pool = MLX::NN::MaxPool2d.new([2, 2])
    avg_pool = MLX::NN::AvgPool2d.new([2, 2])

    assert_nested_close [[[[4.0]]]], max_pool.call(x).to_a
    assert_nested_close [[[[2.5]]]], avg_pool.call(x).to_a
  end

  def test_pool3d_max_and_avg
    x = MLX::Core.array(
      [[[[[1.0], [2.0]], [[3.0], [4.0]]], [[[5.0], [6.0]], [[7.0], [8.0]]]]],
      MLX::Core.float32
    )

    max_pool = MLX::NN::MaxPool3d.new([2, 2, 2])
    avg_pool = MLX::NN::AvgPool3d.new([2, 2, 2])

    assert_nested_close [[[[[8.0]]]]], max_pool.call(x).to_a
    assert_nested_close [[[[[4.5]]]]], avg_pool.call(x).to_a
  end

  def test_pool_argument_validation
    assert_raises(ArgumentError) { MLX::NN::MaxPool1d.new([2, 2]) }
    assert_raises(ArgumentError) { MLX::NN::AvgPool2d.new([2]) }
    assert_raises(ArgumentError) { MLX::NN::MaxPool3d.new([2, 2]) }
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
