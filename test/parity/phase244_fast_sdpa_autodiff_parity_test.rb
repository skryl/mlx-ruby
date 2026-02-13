# frozen_string_literal: true

require_relative "test_helper"

class Phase244FastSdpaAutodiffParityTest < Minitest::Test
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

  def test_sdpa_vjp_and_grad_shapes
    q = MLX::Core.array([[[[0.1, 0.2], [0.3, 0.4]]]], MLX::Core.float32)
    k = MLX::Core.array([[[[0.2, 0.1], [0.4, 0.3]]]], MLX::Core.float32)
    v = MLX::Core.array([[[[1.0, 2.0], [3.0, 4.0]]]], MLX::Core.float32)
    mask = MLX::Core.array([[[[true, false], [true, true]]]], MLX::Core.bool_)

    fn = ->(qq, kk, vv) { MLX::Core.scaled_dot_product_attention(qq, kk, vv, 1.0, mask) }
    out = fn.call(q, k, v)
    cot = MLX::Core.ones_like(out)
    _, vjps = MLX::Core.vjp(fn, [q, k, v], [cot])
    assert_equal q.shape, vjps[0].shape
    assert_equal k.shape, vjps[1].shape
    assert_equal v.shape, vjps[2].shape

    loss = ->(qq) { MLX::Core.sum(MLX::Core.scaled_dot_product_attention(qq, k, v, 1.0, mask)) }
    grad_q = MLX::Core.grad(loss).call(q)
    assert_equal q.shape, grad_q.shape
  end

  def test_sdpa_grad_with_sliced_query
    q_full = MLX::Core.array([[[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]]], MLX::Core.float32)
    q = MLX::Core.reshape(q_full.__getitem__(0), [1, 1, 3, 2])
    k = MLX::Core.array([[[[0.2, 0.1], [0.4, 0.3], [0.5, 0.7]]]], MLX::Core.float32)
    v = MLX::Core.array([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], MLX::Core.float32)

    grad = MLX::Core.grad(->(qq) { MLX::Core.sum(MLX::Core.scaled_dot_product_attention(qq, k, v, 1.0, nil)) }).call(q)
    assert_equal q.shape, grad.shape
  end
end
