# frozen_string_literal: true

require_relative "test_helper"

class Phase27UnaryMathExtendedTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_sign_negative_square_and_reciprocal
    x = MLX::Core.array([-2.0, -1.0, 0.0, 2.0], MLX::Core.float32)

    assert_nested_close [2.0, 1.0, 0.0, -2.0], MLX::Core.negative(x).to_a
    assert_nested_close [-1.0, -1.0, 0.0, 1.0], MLX::Core.sign(x).to_a
    assert_nested_close [4.0, 1.0, 0.0, 4.0], MLX::Core.square(x).to_a

    y = MLX::Core.array([2.0, 4.0], MLX::Core.float32)
    assert_nested_close [0.5, 0.25], MLX::Core.reciprocal(y).to_a
  end

  def test_trig_hyperbolic_and_sqrt_related_ops
    x = MLX::Core.array([0.0, 1.0], MLX::Core.float32)
    angles = MLX::Core.array([0.0, Math::PI / 4], MLX::Core.float32)

    assert_nested_close [0.0, Math.tan(Math::PI / 4)], MLX::Core.tan(angles).to_a, 1e-4
    assert_nested_close [0.0, Math.sinh(1.0)], MLX::Core.sinh(x).to_a, 1e-4
    assert_nested_close [1.0, Math.cosh(1.0)], MLX::Core.cosh(x).to_a, 1e-4
    assert_nested_close [0.0, Math.tanh(1.0)], MLX::Core.tanh(x).to_a, 1e-4

    assert_nested_close [0.0, 1.0, 2.0, 3.0], MLX::Core.sqrt(MLX::Core.array([0.0, 1.0, 4.0, 9.0], MLX::Core.float32)).to_a
  end

  def test_log1p_and_expm1
    x = MLX::Core.array([0.0, 1.0, 3.0], MLX::Core.float32)

    logged = MLX::Core.log1p(x)
    assert_nested_close [0.0, Math.log(2.0), Math.log(4.0)], logged.to_a, 1e-4
    assert_nested_close [0.0, 1.0, 3.0], MLX::Core.expm1(logged).to_a, 1e-4
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
