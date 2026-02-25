# frozen_string_literal: true

require_relative "../support/test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class CoreApiGapRegressionTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_array_accepts_dtype_keyword_argument
    array = MLX::Core.array([1, 2, 3], dtype: MLX::Core.int32)

    assert_equal :int32, array.dtype.name
    assert_equal [1, 2, 3], array.to_a
  end

  def test_array_rejects_conflicting_dtype_positional_and_keyword_arguments
    error = assert_raises(ArgumentError) do
      MLX::Core.array([1, 2, 3], MLX::Core.float32, dtype: MLX::Core.int32)
    end
    assert_match(/conflicting dtype/i, error.message)
  end

  def test_mean_supports_keepdims_keyword
    matrix = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)

    by_row = MLX::Core.mean(matrix, 1, keepdims: true)
    assert_equal [2, 1], by_row.shape
    assert_nested_close [[1.5], [3.5]], by_row.to_a

    global = MLX::Core.mean(matrix, keepdims: true)
    assert_equal [1, 1], global.shape
    assert_nested_close [[2.5]], global.to_a
  end

  def test_numeric_left_hand_scalar_ops_work_with_arrays
    array = MLX::Core.array([1.0, 2.0], MLX::Core.float32)

    assert_nested_close [2.5, 3.5], (1.5 + array).to_a
    assert_nested_close [1.5, 3.0], (1.5 * array).to_a
    assert_nested_close [1.0, 0.0], (2 - array).to_a
    assert_nested_close [2.0, 1.0], (2 / array).to_a
  end

  def test_array_supports_unary_negation_operator
    array = MLX::Core.array([1.0, -2.0], MLX::Core.float32)
    assert_nested_close [-1.0, 2.0], (-array).to_a
  end

  def test_array_supports_comparison_operators
    array = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    assert_equal [false, false, true], (array > 2.0).to_a
    assert_equal [true, false, false], (array < 2.0).to_a
    assert_equal [true, true, false], (array <= 2.0).to_a
    assert_equal [false, true, true], (array >= 2.0).to_a
  end

  def test_array_accepts_f32_dtype_alias_strings_from_safetensors_tooling
    array = MLX::Core.array([1.0, 2.0], "F32")
    assert_equal :float32, array.dtype.name

    both = MLX::Core.array([3.0, 4.0], "float32", dtype: "F32")
    assert_equal :float32, both.dtype.name
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
