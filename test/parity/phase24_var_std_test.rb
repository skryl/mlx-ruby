# frozen_string_literal: true

require_relative "test_helper"

class Phase24VarStdTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_var_and_std_global
    x = MLX::Core.array([1.0, 2.0, 3.0, 4.0], MLX::Core.float32)

    assert_in_delta 1.25, MLX::Core.var(x).to_a, 1e-5
    assert_in_delta Math.sqrt(1.25), MLX::Core.std(x).to_a, 1e-5
    assert_in_delta(5.0 / 3.0, MLX::Core.var(x, nil, false, 1).to_a, 1e-5)
  end

  def test_var_and_std_with_axis
    x = MLX::Core.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], MLX::Core.float32)

    assert_nested_close [2.25, 2.25, 2.25], MLX::Core.var(x, 0).to_a
    assert_nested_close [Math.sqrt(2.0 / 3.0), Math.sqrt(2.0 / 3.0)], MLX::Core.std(x, 1).to_a
    assert_nested_close [1.0, 1.0], MLX::Core.std(x, 1, false, 1).to_a
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
