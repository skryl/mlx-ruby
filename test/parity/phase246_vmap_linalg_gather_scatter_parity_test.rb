# frozen_string_literal: true

require_relative "test_helper"

class Phase246VmapLinalgGatherScatterParityTest < Minitest::Test
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

  def test_vmap_matmul_svd_inverse
    mats = MLX::Core.array([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 1.0], [0.0, 1.0]]], MLX::Core.float32)
    vec = MLX::Core.array([[1.0], [1.0]], MLX::Core.float32)

    matmul = MLX::Core.vmap(->(m) { MLX::Core.matmul(m, vec) })
    assert_equal [[[3.0], [7.0]], [[3.0], [1.0]]], matmul.call(mats).to_a

    inverse = MLX::Core.vmap(->(m) { MLX::Core.inv(m) })
    inv = inverse.call(mats).to_a
    assert_in_delta(-2.0, inv[0][0][0], 1e-5)
    assert_in_delta(0.5, inv[1][0][0], 1e-5)

    svd = MLX::Core.vmap(->(m) { u, s, vt = MLX::Core.svd(m); [u, s, vt] })
    u, s, vt = svd.call(mats)
    assert_equal [2, 2, 2], u.shape
    assert_equal [2, 2], s.shape
    assert_equal [2, 2, 2], vt.shape
  end

  def test_vmap_take_and_put_along_axis_scatter_like
    mats = MLX::Core.array([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 1.0], [0.0, 1.0]]], MLX::Core.float32)

    idx_take = MLX::Core.array([[1, 0], [0, 1]], MLX::Core.int32)
    gather_like = MLX::Core.vmap(->(m) { MLX::Core.take_along_axis(m, idx_take, 1) })
    assert_equal [[[2.0, 1.0], [3.0, 4.0]], [[1.0, 2.0], [0.0, 1.0]]], gather_like.call(mats).to_a

    idx_put = MLX::Core.array([[0, 0]], MLX::Core.int32)
    vals_put = MLX::Core.array([[9.0, 9.0]], MLX::Core.float32)
    scatter_like = MLX::Core.vmap(->(m) { MLX::Core.put_along_axis(m, idx_put, vals_put, 0) })
    assert_equal [[[9.0, 9.0], [3.0, 4.0]], [[9.0, 9.0], [0.0, 1.0]]], scatter_like.call(mats).to_a
  end
end
