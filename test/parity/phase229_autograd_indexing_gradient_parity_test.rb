# frozen_string_literal: true

require_relative "test_helper"

class Phase229AutogradIndexingGradientParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_topk_grad
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    grad = MLX::Core.grad(lambda do |v|
      vals, _idx = MLX::Core.topk(v, 2)
      MLX::Core.sum(vals)
    end).call(x)

    assert_equal [0.0, 1.0, 1.0], grad.to_a
  end

  def test_put_along_axis_grad_and_slice_grad
    idx = MLX::Core.array([1], MLX::Core.int32)
    vals = MLX::Core.array([9.0], MLX::Core.float32)
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    put_grad = MLX::Core.grad(lambda do |v|
      out = MLX::Core.put_along_axis(v, idx, vals, 0)
      MLX::Core.sum(out)
    end).call(x)
    assert_equal [1.0, 0.0, 1.0], put_grad.to_a

    slice_grad = MLX::Core.grad(->(v) { MLX::Core.sum(v.__getitem__(1)) }).call(x)
    assert_equal [0.0, 1.0, 0.0], slice_grad.to_a
  end
end
