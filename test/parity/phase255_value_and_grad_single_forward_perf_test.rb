# frozen_string_literal: true

require_relative "test_helper"

class Phase255ValueAndGradSingleForwardPerfTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_value_and_grad_executes_wrapped_forward_once_per_call
    calls = 0
    fn = lambda do |x|
      calls += 1
      MLX::Core.sum(MLX::Core.multiply(x, x))
    end

    wrapped = MLX::Core.value_and_grad(fn)
    value, grad = wrapped.call(MLX::Core.array([2.0, -3.0], MLX::Core.float32))

    assert_equal 1, calls
    assert_in_delta 13.0, value.item, 1e-5
    assert_equal [4.0, -6.0], grad.to_a
  end
end
