# frozen_string_literal: true

require_relative "../test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class DslSplitPlanUnitTest < Minitest::Test
  FakeLoss = Struct.new(:value) do
    def item
      value
    end
  end

  class FakeStep
    attr_reader :calls

    def initialize(losses)
      @losses = losses
      @calls = []
      @index = 0
    end

    def call(*args, **kwargs)
      @calls << [args, kwargs]
      loss = @losses.fetch(@index)
      @index += 1
      loss
    end
  end

  class FakeModel
    attr_reader :step

    def initialize(losses)
      @step = FakeStep.new(losses)
    end

    def train_step(optimizer:, clip_grad_norm:, compile: false, sync: :none, &loss_block)
      _ = [optimizer, clip_grad_norm, compile, sync, loss_block]
      @step
    end
  end

  def test_fit_report_accepts_split_plan_dataset
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) do |x:, y:|
      x.to_f + y.to_f
    end

    plan = MLX::DSL.splits do
      shared collate: :xy
      train [[1.0, 2.0]]
      validation [[3.0, 4.0]]
    end

    report = trainer.fit_report(plan, epochs: 1)

    assert_equal [[], { x: 1.0, y: 2.0 }], model.step.calls[0]
    assert_equal 7.0, report.fetch("epochs")[0].fetch("val_loss")
  end

  def test_split_plan_shared_transform_and_validation_override
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) do |x:, y:|
      x.to_f + y.to_f
    end

    plan = MLX::DSL.splits do
      shared(
        collate: :xy,
        transform: ->(batch) { { x: batch.fetch(:x) + 1.0, y: batch.fetch(:y) } }
      )
      train [[1.0, 2.0]]
      validation [[3.0, 4.0]],
                 transform: ->(batch) { { x: batch.fetch(:x) + 10.0, y: batch.fetch(:y) } }
    end

    report = trainer.fit_report(plan, epochs: 1)

    assert_equal [[], { x: 2.0, y: 2.0 }], model.step.calls[0]
    assert_equal 17.0, report.fetch("epochs")[0].fetch("val_loss")
  end

  def test_trainer_fit_report_with_raw_dataset_remains_compatible
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    report = trainer.fit_report([{ x: 1 }, { x: 2 }], epochs: 1)

    assert_equal 2, report.fetch("epochs")[0].fetch("batches")
  end
end

$LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
