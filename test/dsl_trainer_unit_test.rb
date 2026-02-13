# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class DslTrainerUnitTest < Minitest::Test
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
    attr_reader :checkpoints, :step

    def initialize(losses)
      @step = FakeStep.new(losses)
      @checkpoints = []
    end

    def train_step(optimizer:, clip_grad_norm:, &loss_block)
      _ = [optimizer, clip_grad_norm, loss_block]
      @step
    end

    def save_checkpoint(path, optimizer:, metadata:)
      @checkpoints << { path: path, optimizer: optimizer, metadata: metadata }
      path
    end
  end

  class OneShotDataset
    def initialize(batches)
      @batches = batches
      @consumed = false
    end

    def each
      return if @consumed

      @consumed = true
      @batches.each { |batch| yield batch }
    end
  end

  def test_fit_report_tracks_epoch_metrics_and_callbacks
    model = FakeModel.new([FakeLoss.new(4.0), FakeLoss.new(2.0), FakeLoss.new(3.0), FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: Object.new) { 0 }

    seen = []
    trainer.on(:before_fit) { |ctx| seen << [:before_fit, ctx[:epochs]] }
    trainer.on(:before_epoch) { |ctx| seen << [:before_epoch, ctx[:epoch]] }
    trainer.on(:after_batch) { |ctx| seen << [:after_batch, ctx[:epoch], ctx[:batch_index]] }
    trainer.on(:after_epoch) { |ctx| seen << [:after_epoch, ctx[:epoch], ctx[:epoch_loss]] }
    trainer.on(:after_fit) { |ctx| seen << [:after_fit, ctx[:best_metric]] }

    dataset = [{ x: 1 }, { x: 2 }]
    report = trainer.fit(dataset, epochs: 2, report: true, reduce: :mean)

    assert_equal 4, report.fetch("losses").length
    assert_equal 2, report.fetch("epochs").length
    assert_in_delta 3.0, report.fetch("epochs")[0].fetch("epoch_loss"), 1e-8
    assert_in_delta 2.0, report.fetch("epochs")[1].fetch("epoch_loss"), 1e-8
    assert_in_delta 2.0, report.fetch("best_metric"), 1e-8

    assert_includes seen, [:before_fit, 2]
    assert_includes seen, [:before_epoch, 0]
    assert_includes seen, [:after_batch, 1, 1]
    assert_includes seen, [:after_fit, 2.0]
  end

  def test_fit_can_checkpoint_best_epoch
    model = FakeModel.new([FakeLoss.new(4.0), FakeLoss.new(2.0), FakeLoss.new(3.0), FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    report = trainer.fit(
      [{ x: 1 }, { x: 2 }],
      epochs: 2,
      report: true,
      reduce: :mean,
      checkpoint_path: "/tmp/fake-checkpoint.bin",
      save_best: true,
      metadata: { "run" => "unit" }
    )

    assert_equal 2.0, report.fetch("best_metric")
    assert_equal 2, model.checkpoints.length
    assert_equal 0, model.checkpoints[0].fetch(:metadata).fetch("epoch")
    assert_equal 1, model.checkpoints[1].fetch(:metadata).fetch("epoch")
    assert_equal "unit", model.checkpoints[1].fetch(:metadata).fetch("run")
  end

  def test_fit_returns_losses_array_by_default
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    losses = trainer.fit([{ x: 1 }, { x: 2 }], epochs: 1)
    assert_equal 2, losses.length
  end

  def test_fit_report_supports_custom_monitor_metric
    model = FakeModel.new([FakeLoss.new(4.0), FakeLoss.new(2.0), FakeLoss.new(3.0), FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    report = trainer.fit_report(
      [{ x: 1 }, { x: 2 }],
      epochs: 2,
      reduce: :mean,
      monitor: :peak_loss,
      monitor_mode: :max,
      metric: lambda { |ctx| ctx.fetch(:epoch_losses).max }
    )

    assert_equal "peak_loss", report.fetch("monitor_name")
    assert_equal 4.0, report.fetch("best_metric")
    assert_equal 4.0, report.fetch("epochs")[0].fetch("monitor_value")
    assert_equal 3.0, report.fetch("epochs")[1].fetch("monitor_value")
  end

  def test_checkpoint_path_supports_epoch_metric_template
    model = FakeModel.new([FakeLoss.new(4.0), FakeLoss.new(2.0), FakeLoss.new(3.0), FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    _report = trainer.fit_report(
      [{ x: 1 }, { x: 2 }],
      epochs: 2,
      reduce: :mean,
      checkpoint_path: "/tmp/fake-checkpoint-%{epoch}-%{monitor}.bin",
      save_best: true
    )

    assert_equal 2, model.checkpoints.length
    assert_equal "/tmp/fake-checkpoint-0-3.0.bin", model.checkpoints[0].fetch(:path)
    assert_equal "/tmp/fake-checkpoint-1-2.0.bin", model.checkpoints[1].fetch(:path)
  end

  def test_hook_helper_methods_register_callbacks
    model = FakeModel.new([FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    seen = []
    chained = trainer
      .before_fit { |_ctx| seen << :before_fit }
      .before_epoch { |ctx| seen << [:before_epoch, ctx.fetch(:epoch)] }
      .after_batch { |ctx| seen << [:after_batch, ctx.fetch(:batch_index)] }
      .after_epoch { |ctx| seen << [:after_epoch, ctx.fetch(:epoch)] }
      .checkpoint { |ctx| seen << [:checkpoint, ctx.fetch(:epoch)] }
      .after_fit { |_ctx| seen << :after_fit }

    assert_same trainer, chained

    trainer.fit(
      [{ x: 1 }],
      epochs: 1,
      checkpoint_path: "/tmp/fake-checkpoint-%{epoch}.bin"
    )

    assert_includes seen, :before_fit
    assert_includes seen, [:before_epoch, 0]
    assert_includes seen, [:after_batch, 0]
    assert_includes seen, [:after_epoch, 0]
    assert_includes seen, [:checkpoint, 0]
    assert_includes seen, :after_fit
  end

  def test_fit_report_supports_early_stopping
    model = FakeModel.new([FakeLoss.new(3.0), FakeLoss.new(4.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    report = trainer.fit_report(
      [{ x: 1 }],
      epochs: 5,
      patience: 0
    )

    assert_equal true, report.fetch("stopped_early")
    assert_equal 2, report.fetch("epochs_ran")
    assert_equal 2, report.fetch("epochs").length
    assert_equal 2, report.fetch("losses").length
    assert_equal 3.0, report.fetch("best_metric")
  end

  def test_fit_report_applies_min_delta_to_improvement
    model = FakeModel.new([FakeLoss.new(3.0), FakeLoss.new(2.95), FakeLoss.new(2.7)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    report = trainer.fit_report(
      [{ x: 1 }],
      epochs: 3,
      patience: 0,
      min_delta: 0.1
    )

    assert_equal true, report.fetch("stopped_early")
    assert_equal 2, report.fetch("epochs_ran")
    assert_equal 3.0, report.fetch("best_metric")
    assert_equal false, report.fetch("epochs")[1].fetch("improved")
  end

  def test_fit_supports_dataset_factory_per_epoch
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0), FakeLoss.new(3.0), FakeLoss.new(4.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    calls = []
    reported_size = :unset
    trainer.before_fit { |ctx| reported_size = ctx.fetch(:dataset_size) }
    dataset = lambda do |epoch:|
      calls << epoch
      [{ x: epoch * 10 + 1 }, { x: epoch * 10 + 2 }]
    end

    report = trainer.fit_report(dataset, epochs: 2)

    assert_equal [0, 1], calls
    assert_nil reported_size
    assert_equal 4, report.fetch("losses").length
    assert_equal 2, report.fetch("epochs")[0].fetch("batches")
    assert_equal 2, report.fetch("epochs")[1].fetch("batches")
  end

  def test_fit_report_supports_validation_data_and_val_loss_monitor
    model = FakeModel.new([FakeLoss.new(8.0), FakeLoss.new(9.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { |x:| x.to_f }

    val_calls = []
    validation_data = lambda do |epoch:|
      val_calls << epoch
      [{ x: epoch + 1.0 }, { x: epoch + 3.0 }]
    end

    report = trainer.fit_report(
      [{ x: 0.0 }],
      epochs: 2,
      validation_data: validation_data,
      validation_reduce: :mean,
      monitor: :val_loss,
      monitor_mode: :min
    )

    assert_equal [0, 1], val_calls
    assert_equal 2.0, report.fetch("epochs")[0].fetch("val_loss")
    assert_equal 3.0, report.fetch("epochs")[1].fetch("val_loss")
    assert_equal 2.0, report.fetch("best_metric")
    assert_equal 2.0, report.fetch("epochs")[0].fetch("monitor_value")
    assert_equal 3.0, report.fetch("epochs")[1].fetch("monitor_value")
  end

  def test_fit_supports_train_transform_for_raw_batches
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.fit(
      ["a", "b"],
      epochs: 1,
      train_transform: lambda do |batch, epoch:, batch_index:, kind:, trainer:|
        {
          x: "#{batch}-#{epoch}-#{batch_index}-#{kind}",
          who: trainer.class.name
        }
      end
    )

    assert_equal [[], { x: "a-0-0-train", who: "MLX::DSL::Trainer" }], model.step.calls[0]
    assert_equal [[], { x: "b-0-1-train", who: "MLX::DSL::Trainer" }], model.step.calls[1]
  end

  def test_fit_report_supports_validation_transform
    model = FakeModel.new([FakeLoss.new(7.0), FakeLoss.new(8.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { |x:| x.to_f }

    report = trainer.fit_report(
      [{ x: 0.0 }],
      epochs: 2,
      validation_data: ->(epoch:) { [[1.0], [3.0]] },
      validation_reduce: :mean,
      validation_transform: ->(batch, epoch:) { { x: batch.fetch(0) + epoch } },
      monitor: :val_loss
    )

    assert_equal 2.0, report.fetch("epochs")[0].fetch("val_loss")
    assert_equal 3.0, report.fetch("epochs")[1].fetch("val_loss")
  end

  def test_dataset_factory_supports_mixed_positional_and_keyword_signature
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    seen = []
    dataset = lambda do |epoch, kind:|
      seen << [epoch, kind]
      [{ x: epoch + 1 }]
    end

    report = trainer.fit_report(dataset, epochs: 2)

    assert_equal [[0, :train], [1, :train]], seen
    assert_equal 2, report.fetch("epochs").length
  end

  def test_fit_report_can_disable_batch_loss_retention
    model = FakeModel.new([FakeLoss.new(4.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    report = trainer.fit_report(
      [{ x: 1 }, { x: 2 }],
      epochs: 1,
      keep_losses: false
    )

    assert_equal [], report.fetch("losses")
    assert_equal false, report.fetch("losses_kept")
    assert_equal 3.0, report.fetch("epochs")[0].fetch("epoch_loss")
  end

  def test_strict_data_reuse_raises_for_exhausted_non_rewindable_dataset
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    error = assert_raises(ArgumentError) do
      trainer.fit_report(
        OneShotDataset.new([{ x: 1 }]),
        epochs: 2,
        strict_data_reuse: true
      )
    end

    assert_match(/exhausted/i, error.message)
  end

  def test_strict_data_reuse_allows_dataset_factory
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    report = trainer.fit_report(
      ->(epoch:) { [{ x: epoch }] },
      epochs: 2,
      strict_data_reuse: true
    )

    assert_equal 1, report.fetch("epochs")[0].fetch("batches")
    assert_equal 1, report.fetch("epochs")[1].fetch("batches")
  end

  def test_validation_hook_helper_methods_register_callbacks
    model = FakeModel.new([FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { |x:| x.to_f }

    seen = []
    chained = trainer
      .before_validation { |ctx| seen << [:before_validation, ctx.fetch(:epoch)] }
      .after_validation_batch { |ctx| seen << [:after_validation_batch, ctx.fetch(:batch_index), ctx.fetch(:loss_value)] }
      .after_validation { |ctx| seen << [:after_validation, ctx.fetch(:epoch), ctx.fetch(:val_loss)] }

    assert_same trainer, chained

    trainer.fit_report(
      [{ x: 0.0 }],
      epochs: 1,
      validation_data: [[1.0], [3.0]],
      validation_transform: ->(batch) { { x: batch.fetch(0) } }
    )

    assert_includes seen, [:before_validation, 0]
    assert_includes seen, [:after_validation_batch, 0, 1.0]
    assert_includes seen, [:after_validation_batch, 1, 3.0]
    assert_includes seen, [:after_validation, 0, 2.0]
  end
end

$LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
