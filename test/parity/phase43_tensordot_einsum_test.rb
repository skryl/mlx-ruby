# frozen_string_literal: true

require_relative "test_helper"

class Phase43TensordotEinsumTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_tensordot_with_integer_axes
    a = MLX::Core.array([[1, 2, 3], [4, 5, 6]], MLX::Core.float32)
    b = MLX::Core.array([[1, 2], [3, 4], [5, 6]], MLX::Core.float32)

    out = MLX::Core.tensordot(a, b, 1)
    assert_equal [2, 2], out.shape
    assert_nested_close [[22.0, 28.0], [49.0, 64.0]], out.to_a
  end

  def test_einsum_and_einsum_path
    a = MLX::Core.array([[1, 2, 3], [4, 5, 6]], MLX::Core.float32)
    b = MLX::Core.array([[1, 2], [3, 4], [5, 6]], MLX::Core.float32)

    out = MLX::Core.einsum("ij,jk->ik", a, b)
    assert_nested_close [[22.0, 28.0], [49.0, 64.0]], out.to_a

    m = MLX::Core.array([[1, 2], [3, 4]], MLX::Core.float32)
    trace = MLX::Core.einsum("ii->", m)
    assert_in_delta 5.0, trace.to_a, 1e-5

    path, summary = MLX::Core.einsum_path("ij,jk->ik", a, b)
    assert path.is_a?(Array)
    assert summary.is_a?(String)
    refute_empty summary
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
