# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase152OptimizerGradientTreeInitTest < Minitest::Test
  class CountingOptimizer < MLX::Optimizers::Optimizer
    attr_reader :init_calls

    def initialize(**kwargs)
      @init_calls = 0
      super
    end

    def init_single(_parameter, state)
      @init_calls += 1
      state["init"] = true
      state
    end
  end

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_lazy_init_uses_gradient_tree_not_full_parameter_tree
    opt = CountingOptimizer.new(learning_rate: 0.1)

    gradients = { "w" => MLX::Core.array([1.0], MLX::Core.float32) }
    parameters = {
      "w" => MLX::Core.array([2.0], MLX::Core.float32),
      "unused" => MLX::Core.array([99.0], MLX::Core.float32)
    }

    output = opt.apply_gradients(gradients, parameters)

    assert_equal 1, opt.init_calls
    refute opt.state.key?("unused")
    assert_equal ["w"], output.keys
  end

  def test_assigning_state_forces_reinitialization_on_next_apply
    opt = CountingOptimizer.new(learning_rate: 0.1)

    gradients = { "w" => MLX::Core.array([1.0], MLX::Core.float32) }
    parameters = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    opt.apply_gradients(gradients, parameters)
    assert_equal 1, opt.init_calls

    opt.state = { "step" => 0, "learning_rate" => 0.1 }
    opt.apply_gradients(gradients, parameters)

    assert_equal 2, opt.init_calls
  end
end
