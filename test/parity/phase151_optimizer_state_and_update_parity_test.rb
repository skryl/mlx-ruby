# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase151OptimizerStateAndUpdateParityTest < Minitest::Test
  class CountingOptimizer < MLX::Optimizers::Optimizer
    attr_reader :init_calls

    def initialize(**kwargs)
      @init_calls = 0
      super
    end

    def init_single(_parameter, state)
      @init_calls += 1
      state["initialized"] = true
      state
    end
  end

  class DummyModel
    attr_reader :params, :last_update_strict

    def initialize(params)
      @params = params
      @last_update_strict = nil
    end

    def parameters
      @params
    end

    def update(parameters, strict: true)
      @last_update_strict = strict
      @params = parameters
      self
    end
  end

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_init_single_is_called_once_per_parameter_leaf
    opt = CountingOptimizer.new(learning_rate: 0.1)

    params = {
      "a" => MLX::Core.array([1.0], MLX::Core.float32),
      "nested" => { "b" => MLX::Core.array([2.0], MLX::Core.float32) }
    }
    grads = {
      "a" => MLX::Core.array([0.5], MLX::Core.float32),
      "nested" => { "b" => MLX::Core.array([1.0], MLX::Core.float32) }
    }

    opt.apply_gradients(grads, params)

    assert_equal 2, opt.init_calls
    assert_equal true, opt.state.fetch("a").fetch("initialized")
    assert_equal true, opt.state.fetch("nested").fetch("b").fetch("initialized")

    opt.apply_gradients(grads, params)
    assert_equal 2, opt.init_calls
  end

  def test_update_calls_model_update_with_optimizer_output
    opt = MLX::Optimizers::Optimizer.new(learning_rate: 0.25)
    model = DummyModel.new("w" => MLX::Core.array([4.0], MLX::Core.float32))
    grads = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    out = opt.update(model, grads)

    assert_same model, out
    assert_equal true, model.last_update_strict
    assert_equal 1, opt.step
    assert_nested_close [3.5], model.parameters.fetch("w").to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal shape_signature(expected), shape_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |item| flatten(item) }
  end

  def shape_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |item| shape_signature(item) })]
  end
end
