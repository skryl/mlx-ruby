# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class DslTrainStepUnitTest < Minitest::Test
  class FakeOptimizer
    attr_reader :updates

    def initialize
      @updates = []
    end

    def update(model, grads)
      @updates << { model: model, grads: grads }
    end
  end

  def test_hook_helper_methods_register_callbacks
    with_stubbed_value_and_grad do
      model = Object.new
      optimizer = FakeOptimizer.new
      step = MLX::DSL::TrainStep.new(
        model,
        optimizer: optimizer,
        clip_grad_norm: nil,
        loss_block: ->(**_kwargs) { 1.25 }
      )

      seen = []
      chained = step
        .before_step { |ctx| seen << [:before_step, ctx.fetch(:step)] }
        .after_backward { |ctx| seen << [:after_backward, ctx.fetch(:loss)] }
        .after_step { |ctx| seen << [:after_step, ctx.fetch(:step)] }

      assert_same step, chained

      loss = step.call(x: 1)

      assert_equal 1.25, loss
      assert_equal 1, optimizer.updates.length
      assert_includes seen, [:before_step, 0]
      assert_includes seen, [:after_backward, 1.25]
      assert_includes seen, [:after_step, 0]
    end
  end

  private

  def with_stubbed_value_and_grad
    nn_singleton = class << MLX::NN
      self
    end

    nn_singleton.alias_method(:__dsl_original_value_and_grad, :value_and_grad)
    nn_singleton.define_method(:value_and_grad) do |_model, fn|
      lambda do |*args, **kwargs|
        [fn.call(*args, **kwargs), { "weight" => 0.5 }]
      end
    end

    yield
  ensure
    nn_singleton.alias_method(:value_and_grad, :__dsl_original_value_and_grad)
    nn_singleton.remove_method(:__dsl_original_value_and_grad)
  end
end

$LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
