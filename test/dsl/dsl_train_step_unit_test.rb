# frozen_string_literal: true

require_relative "../test_helper"

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

  def test_hooks_support_priority_every_once_and_if_predicate
    with_stubbed_value_and_grad do
      model = Object.new
      optimizer = FakeOptimizer.new
      step = MLX::DSL::TrainStep.new(
        model,
        optimizer: optimizer,
        clip_grad_norm: nil,
        loss_block: ->(**_kwargs) { 1.25 }
      )

      ordering = []
      every_calls = 0
      once_calls = 0
      conditional_calls = 0

      step.on(:after_step, priority: 10) { ordering << :late }
      step.on(:after_step, priority: -10) { ordering << :early }
      step.on(:after_step, every: 2) { every_calls += 1 }
      step.on(:after_step, once: true) { once_calls += 1 }
      step.on(:after_step, if: ->(ctx) { ctx.fetch(:step) == 1 }) { conditional_calls += 1 }

      step.call(x: 1)
      step.call(x: 2)
      step.call(x: 3)

      assert_equal [[:early, :late], [:early, :late], [:early, :late]], ordering.each_slice(2).to_a
      assert_equal 2, every_calls
      assert_equal 1, once_calls
      assert_equal 1, conditional_calls
    end
  end

  def test_compile_option_wraps_value_and_grad
    with_stubbed_value_and_grad do
      with_stubbed_core_compile do |calls|
        model = Object.new
        optimizer = FakeOptimizer.new
        step = MLX::DSL::TrainStep.new(
          model,
          optimizer: optimizer,
          clip_grad_norm: nil,
          compile: { shapeless: true },
          loss_block: ->(**_kwargs) { 1.25 }
        )

        loss = step.call(x: 1)
        assert_equal 1.25, loss
        assert_equal 1, calls.length
        assert_equal true, calls[0].fetch(:shapeless)
      end
    end
  end

  def test_sync_step_calls_core_eval
    with_stubbed_value_and_grad do
      with_stubbed_core_eval do |calls|
        model = Object.new
        optimizer = FakeOptimizer.new
        step = MLX::DSL::TrainStep.new(
          model,
          optimizer: optimizer,
          clip_grad_norm: nil,
          sync: :step,
          loss_block: ->(**_kwargs) { 1.25 }
        )

        step.call(x: 1)
        step.call(x: 2)
        assert_equal 2, calls.length
      end
    end
  end

  def test_sync_mode_validation_rejects_epoch_for_train_step
    with_stubbed_value_and_grad do
      error = assert_raises(ArgumentError) do
        MLX::DSL::TrainStep.new(
          Object.new,
          optimizer: FakeOptimizer.new,
          clip_grad_norm: nil,
          sync: :epoch,
          loss_block: ->(**_kwargs) { 1.25 }
        )
      end
      assert_match(/sync/i, error.message)
    end
  end

  def test_compile_option_validation_rejects_unknown_keys
    with_stubbed_value_and_grad do
      error = assert_raises(ArgumentError) do
        MLX::DSL::TrainStep.new(
          Object.new,
          optimizer: FakeOptimizer.new,
          clip_grad_norm: nil,
          compile: { unsupported: true },
          loss_block: ->(**_kwargs) { 1.25 }
        )
      end
      assert_match(/compile option/i, error.message)
    end
  end

  def test_compile_option_validation_rejects_invalid_type
    with_stubbed_value_and_grad do
      error = assert_raises(ArgumentError) do
        MLX::DSL::TrainStep.new(
          Object.new,
          optimizer: FakeOptimizer.new,
          clip_grad_norm: nil,
          compile: :yes,
          loss_block: ->(**_kwargs) { 1.25 }
        )
      end
      assert_match(/compile must/i, error.message)
    end
  end

  private

  def with_stubbed_value_and_grad
    nn_singleton = class << MLX::NN
      self
    end

    nn_singleton.alias_method(:__dsl_original_value_and_grad, :value_and_grad)
    if nn_singleton.instance_methods(false).include?(:value_and_grad)
      nn_singleton.remove_method(:value_and_grad)
    end
    nn_singleton.define_method(:value_and_grad) do |_model, fn|
      lambda do |*args, **kwargs|
        [fn.call(*args, **kwargs), { "weight" => 0.5 }]
      end
    end

    yield
  ensure
    if nn_singleton.instance_methods(false).include?(:value_and_grad)
      nn_singleton.remove_method(:value_and_grad)
    end
    nn_singleton.alias_method(:value_and_grad, :__dsl_original_value_and_grad)
    nn_singleton.remove_method(:__dsl_original_value_and_grad)
  end

  def with_stubbed_core_compile
    core_singleton = class << MLX::Core
      self
    end

    had_compile = core_singleton.instance_methods(false).include?(:compile)
    core_singleton.alias_method(:__dsl_original_compile, :compile) if had_compile

    calls = []
    core_singleton.remove_method(:compile) if had_compile
    core_singleton.define_method(:compile) do |fn, inputs = nil, outputs = nil, shapeless = false|
      calls << { inputs: inputs, outputs: outputs, shapeless: shapeless }
      lambda do |*args, **kwargs|
        fn.call(*args, **kwargs)
      end
    end

    yield calls
  ensure
    if had_compile
      core_singleton.remove_method(:compile) if core_singleton.instance_methods(false).include?(:compile)
      core_singleton.alias_method(:compile, :__dsl_original_compile)
      core_singleton.remove_method(:__dsl_original_compile)
    else
      core_singleton.remove_method(:compile) if core_singleton.instance_methods(false).include?(:compile)
    end
  end

  def with_stubbed_core_eval
    core_singleton = class << MLX::Core
      self
    end

    had_eval = core_singleton.instance_methods(false).include?(:eval)
    core_singleton.alias_method(:__dsl_original_eval, :eval) if had_eval

    calls = []
    core_singleton.remove_method(:eval) if had_eval
    core_singleton.define_method(:eval) do |*args|
      calls << args
      nil
    end

    yield calls
  ensure
    if had_eval
      core_singleton.remove_method(:eval) if core_singleton.instance_methods(false).include?(:eval)
      core_singleton.alias_method(:eval, :__dsl_original_eval)
      core_singleton.remove_method(:__dsl_original_eval)
    else
      core_singleton.remove_method(:eval) if core_singleton.instance_methods(false).include?(:eval)
    end
  end
end

$LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
