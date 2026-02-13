# frozen_string_literal: true

require_relative "test_helper"

class Phase54FftRealShiftTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_rfft_irfft_roundtrip
    x = MLX::Core.array([1.0, 2.0, 3.0, 4.0], MLX::Core.float32)

    y = MLX::Core.rfft(x)
    assert_equal [3], y.shape

    z = MLX::Core.irfft(y, 4)
    assert_equal [4], z.shape
    assert_nested_close x.to_a, z.to_a
  end

  def test_rfft2_rfftn_and_inverse
    x = MLX::Core.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], MLX::Core.float32)

    y2 = MLX::Core.rfft2(x)
    assert_equal [2, 3], y2.shape
    z2 = MLX::Core.irfft2(y2, [2, 4], [-2, -1])
    assert_nested_close x.to_a, z2.to_a

    yn = MLX::Core.rfftn(x)
    assert_equal [2, 3], yn.shape
    zn = MLX::Core.irfftn(yn, [2, 4], [-2, -1])
    assert_nested_close x.to_a, zn.to_a
  end

  def test_fftshift_and_ifftshift
    x = MLX::Core.array([0, 1, 2, 3], MLX::Core.float32)

    shifted = MLX::Core.fftshift(x)
    assert_equal [2.0, 3.0, 0.0, 1.0], shifted.to_a

    unshifted = MLX::Core.ifftshift(shifted)
    assert_equal [0.0, 1.0, 2.0, 3.0], unshifted.to_a
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
