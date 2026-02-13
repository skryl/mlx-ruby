# frozen_string_literal: true

require_relative "test_helper"

class Phase93CustomFunctionVmapIntegrationTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_custom_function_vmap_override_is_used
    fun = MLX::Core.custom_function do |x|
      MLX::Core.square(x)
    end
    fun.vmap do |inputs, axes|
      [MLX::Core.add(inputs[0], 1.0), [axes[0]]]
    end

    vmapped = MLX::Core.vmap(fun, 0, 0)
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    out = vmapped.call(x)
    assert_equal [[2.0, 3.0], [4.0, 5.0]], out.to_a
  end
end
