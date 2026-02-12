# frozen_string_literal: true

require_relative "test_helper"

class Phase28InverseTrigConversionTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_inverse_trig_ops
    x = MLX::Core.array([-1.0, 0.0, 1.0], MLX::Core.float32)

    assert_nested_close [-Math::PI / 2, 0.0, Math::PI / 2], MLX::Core.arcsin(x).to_a, 1e-4
    assert_nested_close [Math::PI, Math::PI / 2, 0.0], MLX::Core.arccos(x).to_a, 1e-4
    assert_nested_close [-Math::PI / 4, 0.0, Math::PI / 4], MLX::Core.arctan(x).to_a, 1e-4

    y = MLX::Core.array([0.0, 1.0], MLX::Core.float32)
    x2 = MLX::Core.array([1.0, 1.0], MLX::Core.float32)
    assert_nested_close [0.0, Math::PI / 4], MLX::Core.arctan2(y, x2).to_a, 1e-4
  end

  def test_degrees_and_radians
    radians = MLX::Core.array([0.0, Math::PI / 2, Math::PI], MLX::Core.float32)
    degrees = MLX::Core.array([0.0, 90.0, 180.0], MLX::Core.float32)

    assert_nested_close [0.0, 90.0, 180.0], MLX::Core.degrees(radians).to_a, 1e-4
    assert_nested_close [0.0, Math::PI / 2, Math::PI], MLX::Core.radians(degrees).to_a, 1e-4
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
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
