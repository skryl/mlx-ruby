# frozen_string_literal: true

require_relative "test_helper"

class Phase227AutogradScatterOverwriteParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_put_along_axis_overwrite_vjp_matches_expected_masking
    idx = MLX::Core.array([1], MLX::Core.int32)
    src = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    updates = MLX::Core.array([7.0], MLX::Core.float32)

    grad_src = MLX::Core.grad(lambda do |x|
      out = MLX::Core.put_along_axis(x, idx, updates, 0)
      MLX::Core.sum(out)
    end).call(src)
    assert_equal [1.0, 0.0, 1.0], grad_src.to_a

    grad_updates = MLX::Core.grad(lambda do |u|
      out = MLX::Core.put_along_axis(src, idx, u, 0)
      MLX::Core.sum(out)
    end).call(updates)
    assert_equal [1.0], grad_updates.to_a
  end
end
