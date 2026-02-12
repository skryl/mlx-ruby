# frozen_string_literal: true

require_relative "test_helper"

class Phase53FftComplexTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_fft_ifft_roundtrip
    x = MLX::Core.array([1.0, 2.0, 3.0, 4.0], MLX::Core.float32)

    y = MLX::Core.fft(x)
    assert_equal [4], y.shape

    z = MLX::Core.ifft(y)
    assert_equal [4], z.shape

    assert_nested_close x.to_a, MLX::Core.real(z).to_a
  end

  def test_fft2_and_nd_transforms
    x = MLX::Core.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], MLX::Core.float32)

    y2 = MLX::Core.fft2(x)
    assert_equal [2, 3], y2.shape
    z2 = MLX::Core.ifft2(y2)
    assert_nested_close x.to_a, MLX::Core.real(z2).to_a

    yn = MLX::Core.fftn(x)
    assert_equal [2, 3], yn.shape
    zn = MLX::Core.ifftn(yn)
    assert_nested_close x.to_a, MLX::Core.real(zn).to_a
  end

  def test_fftn_requires_axes_when_shape_given
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)

    assert_raises(ArgumentError) { MLX::Core.fftn(x, [2, 2]) }
    assert_raises(ArgumentError) { MLX::Core.ifftn(x, [2, 2]) }
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

    value.flat_map { |v| flatten(v) }
  end

  def structure_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |v| structure_signature(v) })]
  end
end
