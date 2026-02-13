# frozen_string_literal: true

require_relative "test_helper"

class Phase232CompileShapelessBaselineParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_shapeless_compile_accepts_varying_input_shapes
    compiled = MLX::Core.compile(->(x) { MLX::Core.add(x, 1.0) }, nil, nil, true)

    a = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    b = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)

    assert_equal [2.0, 3.0], compiled.call(a).to_a
    assert_equal [[2.0, 3.0], [4.0, 5.0]], compiled.call(b).to_a
  end

  def test_shapeless_compile_broadcast_and_reduction
    compiled = MLX::Core.compile(lambda do |x, bias:|
      shifted = MLX::Core.add(x, bias)
      MLX::Core.sum(shifted)
    end, nil, nil, true)

    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    bias = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    assert_in_delta 16.0, compiled.call(x, bias: bias).to_a, 1e-6
  end
end
