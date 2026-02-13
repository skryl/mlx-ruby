# frozen_string_literal: true

require_relative "test_helper"

class Phase25MedianTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_median_global
    x = MLX::Core.array([1.0, 3.0, 2.0, 4.0], MLX::Core.float32)
    assert_in_delta 2.5, MLX::Core.median(x).to_a, 1e-5
  end

  def test_median_axis
    x = MLX::Core.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], MLX::Core.float32)

    assert_nested_close [2.5, 1.5, 3.5], MLX::Core.median(x, 0).to_a

    kept = MLX::Core.median(x, 0, true)
    assert_equal [1, 3], kept.shape
    assert_nested_close [[2.5, 1.5, 3.5]], kept.to_a
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
