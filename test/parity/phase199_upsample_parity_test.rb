# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase199UpsampleParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_upsample_nearest_integer_scale
    x = MLX::Core.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], MLX::Core.float32)
    out = MLX::NN.upsample_nearest(x, [2.0, 2.0])
    assert_nested_close(
      [[[[1.0], [1.0], [2.0], [2.0]], [[1.0], [1.0], [2.0], [2.0]], [[3.0], [3.0], [4.0], [4.0]], [[3.0], [3.0], [4.0], [4.0]]]],
      out.to_a
    )
  end

  def test_upsample_module_validation_and_mode_dispatch
    assert_raises(ArgumentError) { MLX::NN::Upsample.new(scale_factor: 2, mode: "bad") }

    up = MLX::NN::Upsample.new(scale_factor: 2, mode: "nearest")
    x = MLX::Core.array([[[1.0], [2.0]]], MLX::Core.float32)
    assert_nested_close [[[1.0], [1.0], [2.0], [2.0]]], up.call(x).to_a

    bad = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    assert_raises(ArgumentError) { up.call(bad) }

    up_tuple = MLX::NN::Upsample.new(scale_factor: [2.0, 2.0], mode: "nearest")
    assert_raises(ArgumentError) { up_tuple.call(x) }
  end

  def test_linear_and_cubic_modes_expand_shape
    x = MLX::Core.array([[[1.0], [2.0], [3.0]]], MLX::Core.float32)

    linear = MLX::NN::Upsample.new(scale_factor: 2, mode: "linear", align_corners: false)
    cubic = MLX::NN::Upsample.new(scale_factor: 2, mode: "cubic", align_corners: false)

    assert_equal [1, 6, 1], linear.call(x).shape
    assert_equal [1, 6, 1], cubic.call(x).shape
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
