# frozen_string_literal: true

require_relative "test_helper"

class Phase230AutogradStructureTypeParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_vjp_over_concatenate_tree_output
    a = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    b = MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    cot = MLX::Core.array([1.0, 1.0, 1.0, 1.0], MLX::Core.float32)

    fun = ->(x, y) { MLX::Core.concatenate([x, y], 0) }

    out, vjps = MLX::Core.vjp(fun, [a, b], [cot])
    assert_equal [1.0, 2.0, 3.0, 4.0], out[0].to_a
    assert_equal [1.0, 1.0], vjps[0].to_a
    assert_equal [1.0, 1.0], vjps[1].to_a
  end

  def test_matmul_jvp
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    w = MLX::Core.array([[2.0, 0.0], [0.0, 2.0]], MLX::Core.float32)
    t = MLX::Core.ones_like(x)

    outs, jvps = MLX::Core.jvp(->(v) { MLX::Core.matmul(v, w) }, [x], [t])
    assert_equal [[2.0, 4.0], [6.0, 8.0]], outs[0].to_a
    assert_equal [[2.0, 2.0], [2.0, 2.0]], jvps[0].to_a
  end
end
