# frozen_string_literal: true

require_relative "test_helper"

class Phase44GemmHelpersTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_addmm_and_gather_mm
    c = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    a = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    b = MLX::Core.array([[5.0, 6.0], [7.0, 8.0]], MLX::Core.float32)

    out = MLX::Core.addmm(c, a, b, 0.5, 2.0)
    assert_nested_close [[11.5, 15.0], [27.5, 33.0]], out.to_a

    gathered = MLX::Core.gather_mm(a, b)
    assert_nested_close [[19.0, 22.0], [43.0, 50.0]], gathered.to_a
  end

  def test_block_masked_mm_and_segmented_mm
    a = MLX::Core.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], MLX::Core.float32)
    b = MLX::Core.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], MLX::Core.float32)

    masked = MLX::Core.block_masked_mm(a, b, 32)
    assert_nested_close [[58.0, 64.0], [139.0, 154.0]], masked.to_a

    segments = MLX::Core.array([[0, 1], [1, 3]], MLX::Core.uint32)
    segmented = MLX::Core.segmented_mm(a, b, segments)
    assert_equal [2, 2, 2], segmented.shape
    assert_nested_close [[[7.0, 8.0], [28.0, 32.0]], [[51.0, 56.0], [111.0, 122.0]]], segmented.to_a
  end

  def test_hadamard_transform
    x = MLX::Core.array([1.0, 2.0, 3.0, 4.0], MLX::Core.float32)
    out = MLX::Core.hadamard_transform(x, 1.0)
    assert_nested_close [10.0, -2.0, -4.0, 0.0], out.to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-4)
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
