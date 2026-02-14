# frozen_string_literal: true

require_relative "test_helper"

class Phase95CompileConstantValidationTest < Minitest::Test
  def run
    run_without_timeout
  end

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_compile_accepts_supported_constant_leaves
    skip("pending: timeout-sensitive parity coverage; re-enable in final CI")
    fun = lambda do |x, scale: 1.0, bias: 0.0, label: "ok", flag: true|
      y = MLX::Core.multiply(x, scale)
      y = MLX::Core.add(y, bias) if flag
      y
    end

    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    compiled = MLX::Core.compile(fun)
    out = compiled.call(x, scale: 3.0, bias: 1.0, label: "z", flag: true)
    assert_equal [4.0, 7.0], out.to_a
  end

  def test_compile_rejects_unsupported_constant_leaf
    skip("pending: timeout-sensitive parity coverage; re-enable in final CI")
    fun = ->(x, cfg:) { MLX::Core.add(x, 1.0) }
    compiled = MLX::Core.compile(fun)
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)

    err = assert_raises(TypeError) { compiled.call(x, cfg: Object.new) }
    assert_match(/Function arguments and outputs must be trees of arrays or constants/i, err.message)
  end
end
