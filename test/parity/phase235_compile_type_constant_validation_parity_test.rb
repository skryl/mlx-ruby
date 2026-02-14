# frozen_string_literal: true

require_relative "test_helper"

class Phase235CompileTypeConstantValidationParityTest < Minitest::Test
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

  def test_compile_rejects_unsupported_input_types
    compiled = MLX::Core.compile(lambda do |x, cfg:|
      MLX::Core.add(x, 1.0) if cfg
      x
    end)
    x = MLX::Core.array([1.0], MLX::Core.float32)

    assert_raises(TypeError) { compiled.call(x, cfg: Object.new) }
  end

  def test_compile_accepts_none_and_scalar_constants
    compiled = MLX::Core.compile(lambda do |x, scale: 1.0, bias: nil, enabled: true|
      y = MLX::Core.multiply(x, scale)
      y = MLX::Core.add(y, bias) unless bias.nil?
      enabled ? y : x
    end)

    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    assert_equal [2.0, 4.0], compiled.call(x, scale: 2.0, bias: nil, enabled: true).to_a
    assert_equal [1.0, 2.0], compiled.call(x, scale: 2.0, bias: nil, enabled: false).to_a
  end
end
