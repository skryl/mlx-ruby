# frozen_string_literal: true

require_relative "test_helper"

class Phase75CustomFunctionSurfaceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_custom_function_callable_surface
    f = MLX::Core.custom_function(->(x) { MLX::Core.square(x) })
    assert_respond_to f, :call
    assert_respond_to f, :vjp
    assert_respond_to f, :jvp
    assert_respond_to f, :vmap

    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    assert_equal [1.0, 4.0, 9.0], f.call(x).to_a
  end

  def test_custom_function_transform_setters_accept_blocks_and_callables
    f = MLX::Core.custom_function { |x| MLX::Core.add(x, 1.0) }

    vjp_impl = ->(_primals, _cotangents, _outputs) { raise "unused in surface test" }
    jvp_impl = ->(_primals, _tangents) { raise "unused in surface test" }
    vmap_impl = ->(_inputs, _axes) { raise "unused in surface test" }

    assert_same vjp_impl, f.vjp(vjp_impl)
    assert_same jvp_impl, f.jvp(jvp_impl)
    assert_same vmap_impl, f.vmap(vmap_impl)

    assert_respond_to f.vjp {}, :call
    assert_respond_to f.jvp {}, :call
    assert_respond_to f.vmap {}, :call
  end
end
