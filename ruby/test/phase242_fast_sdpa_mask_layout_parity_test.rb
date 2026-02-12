# frozen_string_literal: true

require_relative "test_helper"

class Phase242FastSdpaMaskLayoutParityTest < Minitest::Test
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

  def test_sdpa_broadcast_mask_and_promoted_mask_types
    q = MLX::Core.array([[[[0.1, 0.2], [0.3, 0.4]]]], MLX::Core.float32)
    k = MLX::Core.array([[[[0.2, 0.1], [0.4, 0.3]]]], MLX::Core.float32)
    v = MLX::Core.array([[[[1.0, 2.0], [3.0, 4.0]]]], MLX::Core.float32)

    additive_mask = MLX::Core.array([[[[0.0, -1.0e9]]]], MLX::Core.float32)
    out_add = MLX::Core.scaled_dot_product_attention(q, k, v, 1.0, additive_mask)
    assert_equal [1, 1, 2, 2], out_add.shape

    bool_mask = MLX::Core.array([[[[true, false], [true, true]]]], MLX::Core.bool_)
    out_bool = MLX::Core.scaled_dot_product_attention(q, k, v, 1.0, bool_mask)
    assert_equal [1, 1, 2, 2], out_bool.shape
  end

  def test_sdpa_noncontiguous_inputs_and_attention_sinks
    q = MLX::Core.array([[[[0.1, 0.2], [0.3, 0.4]]]], MLX::Core.float32)
    k = MLX::Core.array([[[[0.2, 0.1], [0.4, 0.3]]]], MLX::Core.float32)
    v = MLX::Core.array([[[[1.0, 2.0], [3.0, 4.0]]]], MLX::Core.float32)

    q_nc = MLX::Core.swapaxes(q, -1, -2)
    k_nc = MLX::Core.swapaxes(k, -1, -2)
    v_nc = MLX::Core.swapaxes(v, -1, -2)
    out_nc = MLX::Core.scaled_dot_product_attention(q_nc, k_nc, v_nc, 1.0, nil)
    assert_equal [1, 1, 2, 2], out_nc.shape

    sinks = MLX::Core.array([0.1], MLX::Core.float32)
    out_sinks = MLX::Core.scaled_dot_product_attention(q, k, v, 1.0, nil, sinks)
    assert_equal [1, 1, 2, 2], out_sinks.shape
  end
end
