# frozen_string_literal: true

require_relative "test_helper"

class Phase5CoreOpsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_reshape_transpose_and_reductions
    x = MLX::Core.array([1, 2, 3, 4], MLX::Core.float32)

    matrix = MLX::Core.reshape(x, [2, 2])
    assert_equal [2, 2], matrix.shape
    assert_nested_close [[1.0, 2.0], [3.0, 4.0]], matrix.to_a

    transposed = MLX::Core.transpose(matrix, [1, 0])
    assert_equal [2, 2], transposed.shape
    assert_nested_close [[1.0, 3.0], [2.0, 4.0]], transposed.to_a

    assert_in_delta 10.0, MLX::Core.sum(matrix).item, 1e-5
    assert_in_delta 2.5, MLX::Core.mean(matrix).item, 1e-5

    assert_nested_close [4.0, 6.0], MLX::Core.sum(matrix, 0).to_a
    assert_nested_close [1.5, 3.5], MLX::Core.mean(matrix, 1).to_a
  end

  def test_random_uniform_with_seed
    MLX::Core.random_seed(123)
    a = MLX::Core.random_uniform([2, 3], 0.0, 1.0, MLX::Core.float32)

    MLX::Core.random_seed(123)
    b = MLX::Core.random_uniform([2, 3], 0.0, 1.0, MLX::Core.float32)

    assert_equal [2, 3], a.shape
    assert_equal [2, 3], b.shape
    assert_equal :float32, a.dtype.name
    assert_equal :float32, b.dtype.name

    assert_nested_close a.to_a, b.to_a
    flatten(a.to_a).each do |v|
      assert_operator v, :>=, 0.0
      assert_operator v, :<=, 1.0
    end
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
