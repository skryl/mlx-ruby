# frozen_string_literal: true

require_relative "test_helper"

class Phase67StreamFftTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_fft_family_accepts_stream_or_device_argument
    x = MLX::Core.array([1.0, 2.0, 3.0, 4.0], MLX::Core.float32)
    cpu = MLX::Core.cpu
    stream = MLX::Core.new_stream(cpu)

    f = MLX::Core.fft(x, nil, -1, cpu)
    assert_equal [4], f.shape

    i = MLX::Core.ifft(f, nil, -1, stream)
    assert_equal [4], i.shape

    r = MLX::Core.rfft(x, nil, -1, cpu)
    assert_equal [3], r.shape

    ir = MLX::Core.irfft(r, 4, -1, stream)
    assert_equal [4], ir.shape

    shifted = MLX::Core.fftshift(x, nil, cpu)
    unshifted = MLX::Core.ifftshift(shifted, nil, stream)
    assert_equal [0.0, 1.0, 2.0, 3.0], MLX::Core.ifftshift(MLX::Core.fftshift(MLX::Core.array([0, 1, 2, 3], MLX::Core.float32), nil, cpu), nil, stream).to_a
    assert_equal [4], unshifted.shape
  end
end
