# frozen_string_literal: true

require_relative "test_helper"

class Phase8UnaryLogicalOpsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_unary_math_ops
    x = MLX::Core.array([-1.0, 0.0, 1.0], MLX::Core.float32)

    assert_nested_close [1.0, 0.0, 1.0], MLX::Core.abs(x).to_a
    assert_nested_close [Math.exp(-1.0), 1.0, Math::E], MLX::Core.exp(x).to_a

    y = MLX::Core.array([1.0, Math::E, Math::E**2], MLX::Core.float32)
    assert_nested_close [0.0, 1.0, 2.0], MLX::Core.log(y).to_a

    angles = MLX::Core.array([0.0, Math::PI / 2, Math::PI], MLX::Core.float32)
    assert_nested_close [0.0, 1.0, 0.0], MLX::Core.sin(angles).to_a, 1e-4
    assert_nested_close [1.0, 0.0, -1.0], MLX::Core.cos(angles).to_a, 1e-4
  end

  def test_logical_and_comparison_helpers
    mixed = MLX::Core.array([0.0, 1.0, MLX::Core.inf, MLX::Core.nan], MLX::Core.float32)
    finite = MLX::Core.isfinite(mixed)
    assert_equal :bool_, finite.dtype.name
    assert_equal [true, true, false, false], finite.to_a

    a = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    b = MLX::Core.array([1.0 + 1e-6, 2.0 - 1e-6, 3.0], MLX::Core.float32)
    c = MLX::Core.array([1.1, 2.0, 3.0], MLX::Core.float32)

    assert MLX::Core.allclose(a, b)
    refute MLX::Core.allclose(a, c)
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
