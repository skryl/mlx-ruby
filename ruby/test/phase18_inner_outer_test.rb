# frozen_string_literal: true

require_relative "test_helper"

class Phase18InnerOuterTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_inner_product_for_vectors
    a = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    b = MLX::Core.array([4.0, 5.0, 6.0], MLX::Core.float32)

    assert_in_delta 32.0, MLX::Core.inner(a, b).to_a, 1e-5
  end

  def test_outer_product_for_vectors
    a = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    b = MLX::Core.array([4.0, 5.0, 6.0], MLX::Core.float32)

    assert_nested_close(
      [[4.0, 5.0, 6.0], [8.0, 10.0, 12.0], [12.0, 15.0, 18.0]],
      MLX::Core.outer(a, b).to_a
    )
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
