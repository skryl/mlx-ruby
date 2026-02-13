# frozen_string_literal: true

require_relative "test_helper"

class Phase59FastOpsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_rms_norm_and_layer_norm
    x = MLX::Core.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], MLX::Core.float32)
    w = MLX::Core.array([1.0, 1.0, 1.0], MLX::Core.float32)
    b = MLX::Core.array([0.0, 0.0, 0.0], MLX::Core.float32)

    rms = MLX::Core.rms_norm(x, w, 1e-5)
    assert_equal [2, 3], rms.shape

    ln = MLX::Core.layer_norm(x, w, b, 1e-5)
    assert_equal [2, 3], ln.shape
  end

  def test_rope_and_scaled_dot_product_attention
    rope_in = MLX::Core.array([[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]]], MLX::Core.float32)
    rope_out = MLX::Core.rope(rope_in, 4, false, 10_000.0, 1.0, 0)
    assert_equal [1, 1, 2, 4], rope_out.shape

    q = MLX::Core.array([[[[0.1, 0.2], [0.3, 0.4]]]], MLX::Core.float32)
    k = MLX::Core.array([[[[0.2, 0.1], [0.4, 0.3]]]], MLX::Core.float32)
    v = MLX::Core.array([[[[1.0, 2.0], [3.0, 4.0]]]], MLX::Core.float32)

    out = MLX::Core.scaled_dot_product_attention(q, k, v, 1.0, "causal")
    assert_equal [1, 1, 2, 2], out.shape
  end
end
