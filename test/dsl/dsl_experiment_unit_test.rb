# frozen_string_literal: true

require_relative "../test_helper"
require "tmpdir"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class DslExperimentUnitTest < Minitest::Test
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

    def trainer(optimizer:, clip_grad_norm: nil, compile: false, sync: :none, &loss_block)
      MLX::DSL::Trainer.new(
        model: self,
        optimizer: optimizer,
        clip_grad_norm: clip_grad_norm,
        compile: compile,
        sync: sync,
        &loss_block
      )
    end

    def save_checkpoint(path, optimizer:, metadata:)
      _ = [optimizer, metadata]
      path
    end
  end

  def test_experiment_declaration_runs_and_saves_run_bundle
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0)])

    experiment = MLX::DSL.experiment("unit-exp") do
      model { model }
      optimizer { :opt }
      trainer do |x:|
        x.to_f
      end
      data train: [{ x: 1.0 }, { x: 2.0 }]
      artifacts checkpoint_path: "/tmp/dsl-exp-%{epoch}.bin"
    end

    report = experiment.report(epochs: 1)
    assert_equal 1, report.fetch("epochs_ran")
    assert_equal 2, report.fetch("epochs")[0].fetch("batches")

    Dir.mktmpdir("mlx-dsl-experiment") do |dir|
      bundle_path = File.join(dir, "run_bundle.json")
      saved_path = experiment.save_run_bundle(bundle_path, config: { "seed" => 42 }, epochs: 1)
      assert_equal bundle_path, saved_path
      assert File.file?(bundle_path)
    end
  end

  def test_experiment_supports_explicit_trainer_injection_and_override_precedence
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0)])
    injected = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    experiment = MLX::DSL.experiment do
      model { raise "should not use model section when trainer injected" }
      optimizer { raise "should not use optimizer section when trainer injected" }
      trainer injected
      data train: [{ x: 1 }, { x: 2 }]
    end

    report = experiment.report(epochs: 1, limit: 1)
    assert_equal 1, report.fetch("epochs_ran")
    assert_equal 1, report.fetch("epochs")[0].fetch("batches")
  end
end

$LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
