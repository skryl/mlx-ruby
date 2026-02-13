# frozen_string_literal: true

require_relative "test_helper"

class Phase6CreationOpsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_arange_and_linspace
    ints = MLX::Core.arange(0, 5, 1, MLX::Core.int32)
    assert_equal :int32, ints.dtype.name
    assert_equal [0, 1, 2, 3, 4], ints.to_a

    floats = MLX::Core.linspace(0.0, 1.0, 5, MLX::Core.float32)
    assert_equal :float32, floats.dtype.name
    assert_nested_close [0.0, 0.25, 0.5, 0.75, 1.0], floats.to_a
  end

  def test_zeros_ones_and_astype
    zeros = MLX::Core.zeros([2, 2], MLX::Core.float32)
    ones = MLX::Core.ones([2, 2], MLX::Core.float32)

    assert_nested_close [[0.0, 0.0], [0.0, 0.0]], zeros.to_a
    assert_nested_close [[1.0, 1.0], [1.0, 1.0]], ones.to_a

    ints = MLX::Core.arange(0, 4, 1, MLX::Core.int32)
    casted = MLX::Core.astype(ints, MLX::Core.float32)

    assert_equal :float32, casted.dtype.name
    assert_nested_close [0.0, 1.0, 2.0, 3.0], casted.to_a
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
