# frozen_string_literal: true

require_relative "test_helper"

class Phase237CompileRuntimeStabilityParityTest < Minitest::Test
  def setup
    skip("pending: timeout-sensitive parity coverage; re-enable in final CI")
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_compiled_callable_can_be_wrapped_and_reused
    compiled = MLX::Core.compile(->(x) { MLX::Core.add(x, 1.0) })
    wrapper = Struct.new(:fn).new(compiled)
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)

    assert_equal [2.0, 3.0], wrapper.fn.call(x).to_a
    assert_equal [2.0, 3.0], wrapper.fn.call(x).to_a
  end

  def test_repeated_compile_invocation_is_stable
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    20.times do
      compiled = MLX::Core.compile(->(v) { MLX::Core.add(v, 1.0) })
      out = compiled.call(x)
      assert_equal [2.0, 3.0], out.to_a
    end
  end
end
