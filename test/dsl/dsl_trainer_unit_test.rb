# frozen_string_literal: true

require_relative "../test_helper"
require "tmpdir"

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
    attr_reader :checkpoints, :step, :train_step_kwargs, :load_checkpoint_calls
    attr_accessor :load_checkpoint_payload

    def initialize(losses)
      @step = FakeStep.new(losses)
      @checkpoints = []
      @train_step_kwargs = nil
      @load_checkpoint_calls = []
      @load_checkpoint_payload = nil
    end

    def train_step(optimizer:, clip_grad_norm:, compile: false, sync: :none, &loss_block)
      @train_step_kwargs = {
        optimizer: optimizer,
        clip_grad_norm: clip_grad_norm,
        compile: compile,
        sync: sync
      }
      _ = loss_block
      @step
    end

    def save_checkpoint(path, optimizer:, metadata:)
      @checkpoints << { path: path, optimizer: optimizer, metadata: metadata }
      path
    end

    def load_checkpoint(path, optimizer:, strict: true, format: nil)
      @load_checkpoint_calls << { path: path, optimizer: optimizer, strict: strict, format: format }
      raise ArgumentError, "fake checkpoint payload is not configured" if @load_checkpoint_payload.nil?

      @load_checkpoint_payload
    end
  end

  class LegacyTrainStepModel
    attr_reader :step, :train_step_kwargs

    def initialize(losses)
      @step = FakeStep.new(losses)
      @train_step_kwargs = nil
    end

    def train_step(optimizer:, clip_grad_norm:, &loss_block)
      _ = loss_block
      @train_step_kwargs = {
        optimizer: optimizer,
        clip_grad_norm: clip_grad_norm
      }
      @step
    end
  end

  class RaisingStep
    def initialize(error)
      @error = error
    end

    def call(*args, **kwargs)
      _ = [args, kwargs]
      raise @error.class, @error.message
    end
  end

  class RaisingTrainStepModel
    def initialize(error)
      @step = RaisingStep.new(error)
    end

    def train_step(optimizer:, clip_grad_norm:, compile: false, sync: :none, &loss_block)
      _ = [optimizer, clip_grad_norm, compile, sync, loss_block]
      @step
    end
  end

  class StrictKeywordStep
    attr_reader :calls

    def initialize(losses)
      @losses = losses
      @calls = []
      @index = 0
    end

    def call(x:, y:)
      @calls << [x, y]
      loss = @losses.fetch(@index)
      @index += 1
      loss
    end
  end

  class StrictKeywordModel
    attr_reader :step

    def initialize(losses)
      @step = StrictKeywordStep.new(losses)
    end

    def train_step(optimizer:, clip_grad_norm:, compile: false, sync: :none, &loss_block)
      _ = [optimizer, clip_grad_norm, compile, sync, loss_block]
      @step
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
    assert_equal 2, report.fetch("epochs_target")
    assert_equal 2, report.fetch("epochs_completed")
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

  def test_fit_report_with_uses_registered_fit_preset
    model = FakeModel.new([FakeLoss.new(3.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { |x:| x.to_f + 1.0 }

    trainer.register_fit_preset(
      :quick_val,
      epochs: 2,
      monitor: :val_loss,
      validation_data: [{ x: 5.0 }],
      validation_reduce: :mean
    )

    report = trainer.fit_report_with(:quick_val, [{ x: 0.0 }])

    assert_equal 2, report.fetch("epochs_ran")
    assert_equal "val_loss", report.fetch("monitor_name")
    assert_equal 6.0, report.fetch("epochs")[0].fetch("val_loss")
  end

  def test_fit_presets_merge_with_trainer_defaults_and_call_overrides
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0), FakeLoss.new(3.0), FakeLoss.new(4.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }
      .with_fit_defaults(reduce: :sum, limit: 1)

    trainer.register_fit_preset(:wide, epochs: 2, limit: 2, monitor: :val_loss)

    report = trainer.fit_report_with(
      :wide,
      [{ x: 1 }, { x: 2 }, { x: 3 }],
      epochs: 1,
      limit: 3,
      monitor: :epoch_loss
    )

    assert_equal 1, report.fetch("epochs_ran")
    assert_equal "epoch_loss", report.fetch("monitor_name")
    assert_equal 3, report.fetch("epochs")[0].fetch("batches")
    assert_equal 6.0, report.fetch("epochs")[0].fetch("epoch_loss")
  end

  def test_with_fit_defaults_returns_configured_copy_without_mutating_original
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0), FakeLoss.new(3.0), FakeLoss.new(4.0)])
    base = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }
    configured = base.with_fit_defaults(limit: 1)

    base_report = base.fit_report([{ x: 1 }, { x: 2 }], epochs: 1)
    configured_report = configured.fit_report([{ x: 1 }, { x: 2 }], epochs: 1)

    assert_equal 2, base_report.fetch("epochs")[0].fetch("batches")
    assert_equal 1, configured_report.fetch("epochs")[0].fetch("batches")
  end

  def test_trainer_forwards_compile_and_step_sync_to_train_step
    model = FakeModel.new([FakeLoss.new(1.0)])
    _trainer = MLX::DSL::Trainer.new(
      model: model,
      optimizer: :opt,
      compile: { shapeless: true },
      sync: :step
    ) { 0 }

    refute_nil model.train_step_kwargs
    assert_equal({ shapeless: true }, model.train_step_kwargs.fetch(:compile))
    assert_equal :step, model.train_step_kwargs.fetch(:sync)
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

  def test_fit_task_report_applies_builtin_classification_defaults
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    report = trainer.fit_task_report(:classification, [[10, 20]], epochs: 1)

    assert_equal "epoch_loss", report.fetch("monitor_name")
    assert_equal [[], { x: 10, y: 20 }], model.step.calls[0]
  end

  def test_fit_task_report_allows_overrides_over_task_defaults
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    report = trainer.fit_task_report(
      :classification,
      [[10, 20]],
      epochs: 1,
      collate: { x: 1, y: 0 },
      monitor: :val_loss,
      validation_data: [[2.0, 3.0]],
      validation_collate: :xy
    )

    assert_equal "val_loss", report.fetch("monitor_name")
    assert_equal [[], { x: 20, y: 10 }], model.step.calls[0]
  end

  def test_fit_task_report_supports_language_modeling_default_metric
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    report = trainer.fit_task_report(:language_modeling, [[1, 2]], epochs: 1)

    assert_equal "perplexity", report.fetch("monitor_name")
    assert_in_delta Math.exp(1.0), report.fetch("epochs")[0].fetch("monitor_value"), 1e-8
  end

  def test_fit_task_rejects_unknown_task
    trainer = MLX::DSL::Trainer.new(model: FakeModel.new([FakeLoss.new(1.0)]), optimizer: :opt) { 0 }

    error = assert_raises(ArgumentError) do
      trainer.fit_task(:totally_unknown, [{ x: 1 }], epochs: 1)
    end
    assert_match(/unknown task/i, error.message)
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

  def test_checkpoint_path_supports_next_epoch_template
    model = FakeModel.new([FakeLoss.new(3.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.fit_report(
      [{ x: 1 }],
      epochs: 2,
      checkpoint_path: "/tmp/fake-checkpoint-next-%{next_epoch}.bin"
    )

    assert_equal "/tmp/fake-checkpoint-next-1.bin", model.checkpoints[0].fetch(:path)
    assert_equal "/tmp/fake-checkpoint-next-2.bin", model.checkpoints[1].fetch(:path)
  end

  def test_checkpoint_path_supports_callable_builder_with_context
    model = FakeModel.new([FakeLoss.new(3.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    seen = []
    trainer.fit_report(
      [{ x: 1 }],
      epochs: 2,
      checkpoint_path: lambda do |epoch:, next_epoch:, monitor:, monitor_name:, epoch_loss:, improved:, trainer:, model:, optimizer:|
        seen << [epoch, next_epoch, monitor, monitor_name, epoch_loss, improved, trainer.class.name, model.class.name, optimizer]
        "/tmp/fake-checkpoint-callable-#{epoch}-#{next_epoch}-#{monitor_name}-#{monitor}.bin"
      end
    )

    assert_equal "/tmp/fake-checkpoint-callable-0-1-epoch_loss-3.0.bin", model.checkpoints[0].fetch(:path)
    assert_equal "/tmp/fake-checkpoint-callable-1-2-epoch_loss-2.0.bin", model.checkpoints[1].fetch(:path)
    assert_equal [
      [0, 1, 3.0, "epoch_loss", 3.0, true, "MLX::DSL::Trainer", "DslTrainerUnitTest::FakeModel", :opt],
      [1, 2, 2.0, "epoch_loss", 2.0, true, "MLX::DSL::Trainer", "DslTrainerUnitTest::FakeModel", :opt]
    ], seen
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

  def test_hooks_support_priority_ordering
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    seen = []
    trainer.on(:after_batch, priority: 10) { seen << :late }
    trainer.on(:after_batch, priority: -10) { seen << :early }

    trainer.fit([{ x: 1 }], epochs: 1)

    assert_equal [:early, :late], seen
  end

  def test_hooks_support_every_once_and_if_predicate
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0), FakeLoss.new(3.0), FakeLoss.new(4.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    every_calls = 0
    once_calls = 0
    conditional_calls = 0
    trainer.on(:after_batch, every: 2) { every_calls += 1 }
    trainer.on(:after_batch, once: true) { once_calls += 1 }
    trainer.on(:after_batch, if: ->(ctx) { ctx.fetch(:epoch) == 1 }) { conditional_calls += 1 }

    trainer.fit([{ x: 1 }, { x: 2 }], epochs: 2)

    assert_equal 2, every_calls
    assert_equal 1, once_calls
    assert_equal 2, conditional_calls
  end

  def test_use_hook_pack_registers_hooks_with_options_and_context
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0), FakeLoss.new(3.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    seen = []
    trainer.register_hook_pack(:batch_trace) do |trainer:, every:, store:|
      trainer.after_batch(every: every) do |ctx|
        store << [ctx.fetch(:batch_index), trainer.class.name]
      end
    end
    trainer.use_hook_pack(:batch_trace, every: 2, store: seen)

    trainer.fit([{ x: 1 }, { x: 2 }, { x: 3 }], epochs: 1)

    assert_equal [[0, "MLX::DSL::Trainer"], [2, "MLX::DSL::Trainer"]], seen
  end

  def test_fit_report_supports_registered_named_metric
    model = FakeModel.new([FakeLoss.new(4.0), FakeLoss.new(2.0), FakeLoss.new(3.0), FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }
    trainer.register_metric(:peak_loss) { |ctx| ctx.fetch(:epoch_losses).max }

    report = trainer.fit_report(
      [{ x: 1 }, { x: 2 }],
      epochs: 2,
      reduce: :mean,
      monitor: :peak_loss,
      monitor_mode: :max,
      metric: :peak_loss
    )

    assert_equal "peak_loss", report.fetch("monitor_name")
    assert_equal 4.0, report.fetch("best_metric")
    assert_equal 4.0, report.fetch("epochs")[0].fetch("monitor_value")
    assert_equal 3.0, report.fetch("epochs")[1].fetch("monitor_value")
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

  def test_fit_report_can_resume_from_checkpoint_metadata
    model = FakeModel.new([FakeLoss.new(1.5), FakeLoss.new(1.0)])
    model.load_checkpoint_payload = {
      "format" => "mlx_dsl_checkpoint_v2_native",
      "metadata" => {
        "epoch" => 1,
        "monitor_name" => "epoch_loss",
        "best_metric" => 2.0,
        "stale_epochs" => 0
      }
    }
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    report = trainer.fit_report(
      [{ x: 1 }],
      epochs: 4,
      resume_from: "/tmp/fake-resume.bin"
    )

    assert_equal 1, model.load_checkpoint_calls.length
    assert_equal "/tmp/fake-resume.bin", model.load_checkpoint_calls[0].fetch(:path)
    assert_equal :opt, model.load_checkpoint_calls[0].fetch(:optimizer)
    assert_equal [2, 3], report.fetch("epochs").map { |row| row.fetch("epoch") }
    assert_equal 2, report.fetch("epochs_ran")
    assert_equal 4, report.fetch("epochs_target")
    assert_equal 4, report.fetch("epochs_completed")
    assert_equal 1.0, report.fetch("best_metric")
    assert_equal "/tmp/fake-resume.bin", report.fetch("resume_from")
    assert_equal 1, report.fetch("resumed_from_epoch")
  end

  def test_fit_report_resume_restores_stale_epochs_for_early_stopping
    model = FakeModel.new([FakeLoss.new(3.1)])
    model.load_checkpoint_payload = {
      "metadata" => {
        "epoch" => 0,
        "monitor_name" => "epoch_loss",
        "best_metric" => 3.0,
        "stale_epochs" => 1
      }
    }
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    report = trainer.fit_report(
      [{ x: 1 }],
      epochs: 3,
      patience: 1,
      resume_from: "/tmp/fake-resume.bin"
    )

    assert_equal true, report.fetch("stopped_early")
    assert_equal 1, report.fetch("epochs_ran")
    assert_equal 2, report.fetch("epochs")[0].fetch("stale_epochs")
  end

  def test_fit_report_rejects_resume_checkpoint_monitor_name_mismatch
    model = FakeModel.new([])
    model.load_checkpoint_payload = {
      "metadata" => {
        "epoch" => 0,
        "monitor_name" => "val_loss",
        "best_metric" => 1.0
      }
    }
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    error = assert_raises(ArgumentError) do
      trainer.fit_report(
        [{ x: 1 }],
        epochs: 1,
        monitor: :epoch_loss,
        resume_from: "/tmp/fake-resume.bin"
      )
    end

    assert_match(/monitor_name/i, error.message)
  end

  def test_run_bundle_includes_report_config_and_checkpoint_snapshot
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }
    report = trainer.fit_report(
      [{ x: 1 }],
      epochs: 1,
      checkpoint_path: "/tmp/fake-checkpoint-%{epoch}.bin"
    )

    bundle = trainer.run_bundle(report: report, config: { "seed" => 7 })

    assert_equal "mlx_dsl_run_bundle_v1", bundle.fetch("format")
    assert_equal report, bundle.fetch("report")
    assert_equal({ "seed" => 7 }, bundle.fetch("config"))
    checkpoint = bundle.fetch("checkpoint")
    assert_equal 0, checkpoint.fetch("metadata").fetch("epoch")
    assert_equal report.fetch("monitor_name"), checkpoint.fetch("metadata").fetch("monitor_name")
  end

  def test_resume_payload_from_bundle_supports_resume_flow
    model_a = FakeModel.new([FakeLoss.new(1.0)])
    trainer_a = MLX::DSL::Trainer.new(model: model_a, optimizer: :opt) { 0 }
    report_a = trainer_a.fit_report(
      [{ x: 1 }],
      epochs: 1,
      checkpoint_path: "/tmp/fake-checkpoint-%{epoch}.bin"
    )

    Dir.mktmpdir("mlx-dsl-run-bundle") do |dir|
      bundle_path = File.join(dir, "run_bundle.json")
      trainer_a.save_run_bundle(bundle_path, report: report_a, config: { "run" => "unit" })

      payload = trainer_a.resume_payload_from_bundle(bundle_path)

      model_b = FakeModel.new([FakeLoss.new(0.8)])
      trainer_b = MLX::DSL::Trainer.new(model: model_b, optimizer: :opt) { 0 }
      report_b = trainer_b.fit_report(
        [{ x: 1 }],
        epochs: 2,
        resume_from: payload
      )

      assert_equal [1], report_b.fetch("epochs").map { |row| row.fetch("epoch") }
      assert_equal 1, report_b.fetch("epochs_ran")
      assert_equal 0, payload.fetch("metadata").fetch("epoch")
      assert_equal "epoch_loss", payload.fetch("metadata").fetch("monitor_name")
    end
  end

  def test_artifact_policy_applies_checkpoint_strategy_and_retention
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0), FakeLoss.new(3.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }
    trainer.artifact_policy(
      checkpoint: { path: "/tmp/policy-checkpoint-%{epoch}.bin", strategy: :latest },
      retention: { keep_last_n: 2 }
    )

    report = trainer.fit_report([{ x: 1 }], epochs: 3)

    assert_equal 3, report.fetch("epochs_ran")
    assert_equal 3, model.checkpoints.length
    history = trainer.checkpoint_history
    assert_equal 2, history.length
    assert_equal ["/tmp/policy-checkpoint-1.bin", "/tmp/policy-checkpoint-2.bin"], history.map { |row| row.fetch("path") }
  end

  def test_artifact_policy_resume_latest_uses_retained_checkpoint_metadata
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0), FakeLoss.new(3.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }
    trainer.artifact_policy(
      checkpoint: { path: "/tmp/policy-resume-%{epoch}.bin", strategy: :latest },
      resume: :latest
    )

    trainer.fit_report([{ x: 1 }], epochs: 2)
    report = trainer.fit_report([{ x: 1 }], epochs: 3)

    assert_equal [2], report.fetch("epochs").map { |row| row.fetch("epoch") }
    assert_equal [], model.load_checkpoint_calls
  end

  def test_artifact_policy_auto_saves_run_bundle_and_includes_policy_metadata
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    Dir.mktmpdir("mlx-dsl-policy-bundle") do |dir|
      bundle_path = File.join(dir, "auto_bundle.json")
      trainer.artifact_policy(
        run_bundle: {
          enabled: true,
          path: bundle_path,
          config: { "source" => "policy" }
        }
      )

      report = trainer.fit_report([{ x: 1 }], epochs: 1)

      assert_equal bundle_path, report.fetch("run_bundle_path")
      assert_equal true, report.fetch("artifact_policy").fetch("run_bundle").fetch("enabled")
      assert File.file?(bundle_path)

      bundle = JSON.parse(File.binread(bundle_path))
      assert_equal "policy", bundle.fetch("config").fetch("source")
    end
  end

  def test_fit_report_accepts_run_bundle_hash_as_resume_source
    model_a = FakeModel.new([FakeLoss.new(1.0)])
    trainer_a = MLX::DSL::Trainer.new(model: model_a, optimizer: :opt) { 0 }
    report_a = trainer_a.fit_report(
      [{ x: 1 }],
      epochs: 1,
      checkpoint_path: "/tmp/fake-checkpoint-%{epoch}.bin"
    )
    bundle = trainer_a.run_bundle(report: report_a, config: { "run" => "unit" })

    model_b = FakeModel.new([FakeLoss.new(0.8)])
    trainer_b = MLX::DSL::Trainer.new(model: model_b, optimizer: :opt) { 0 }
    report_b = trainer_b.fit_report(
      [{ x: 1 }],
      epochs: 2,
      resume_from: bundle
    )

    assert_equal [], model_b.load_checkpoint_calls
    assert_equal [1], report_b.fetch("epochs").map { |row| row.fetch("epoch") }
    assert_nil report_b.fetch("resume_from")
    assert_equal 0, report_b.fetch("resumed_from_epoch")
  end

  def test_fit_report_accepts_run_bundle_path_as_resume_source
    model_a = FakeModel.new([FakeLoss.new(1.0)])
    trainer_a = MLX::DSL::Trainer.new(model: model_a, optimizer: :opt) { 0 }
    report_a = trainer_a.fit_report(
      [{ x: 1 }],
      epochs: 1,
      checkpoint_path: "/tmp/fake-checkpoint-%{epoch}.bin"
    )

    Dir.mktmpdir("mlx-dsl-run-bundle-resume") do |dir|
      bundle_path = File.join(dir, "run_bundle.json")
      trainer_a.save_run_bundle(bundle_path, report: report_a, config: { "run" => "unit" })

      model_b = FakeModel.new([FakeLoss.new(0.75)])
      trainer_b = MLX::DSL::Trainer.new(model: model_b, optimizer: :opt) { 0 }
      report_b = trainer_b.fit_report(
        [{ x: 1 }],
        epochs: 2,
        resume_from: bundle_path
      )

      assert_equal [], model_b.load_checkpoint_calls
      assert_equal [1], report_b.fetch("epochs").map { |row| row.fetch("epoch") }
      assert_equal bundle_path, report_b.fetch("resume_from")
      assert_equal 0, report_b.fetch("resumed_from_epoch")
    end
  end

  def test_fit_report_accepts_inline_resume_payload_hash
    model = FakeModel.new([FakeLoss.new(1.5)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    report = trainer.fit_report(
      [{ x: 1 }],
      epochs: 2,
      resume_from: {
        "metadata" => {
          "epoch" => 0,
          "monitor_name" => "epoch_loss",
          "best_metric" => 2.0,
          "stale_epochs" => 0
        }
      }
    )

    assert_equal [], model.load_checkpoint_calls
    assert_equal [1], report.fetch("epochs").map { |row| row.fetch("epoch") }
    assert_equal 1, report.fetch("epochs_ran")
    assert_nil report.fetch("resume_from")
    assert_equal 0, report.fetch("resumed_from_epoch")
  end

  def test_fit_report_accepts_callable_resume_loader_with_context
    model = FakeModel.new([FakeLoss.new(1.25)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    seen = []
    resume_loader = lambda do |trainer:, model:, optimizer:, monitor_name:|
      seen << [trainer.class.name, model.class.name, optimizer, monitor_name]
      {
        "metadata" => {
          "epoch" => 0,
          "monitor_name" => monitor_name,
          "best_metric" => 2.0,
          "stale_epochs" => 0
        }
      }
    end

    report = trainer.fit_report(
      [{ x: 1 }],
      epochs: 2,
      resume_from: resume_loader
    )

    assert_equal [["MLX::DSL::Trainer", "DslTrainerUnitTest::FakeModel", :opt, "epoch_loss"]], seen
    assert_equal [], model.load_checkpoint_calls
    assert_equal [1], report.fetch("epochs").map { |row| row.fetch("epoch") }
    assert_equal 1.25, report.fetch("best_metric")
  end

  def test_epoch_sync_calls_core_eval_once_per_epoch
    model = FakeModel.new([FakeLoss.new(3.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt, sync: :epoch) { 0 }

    with_stubbed_core_eval do |calls|
      report = trainer.fit_report([{ x: 1 }], epochs: 2)
      assert_equal 2, report.fetch("epochs_ran")
      assert_equal 2, calls.length
    end
  end

  def test_trainer_compile_and_sync_are_backward_compatible_with_legacy_train_step_signature
    model = LegacyTrainStepModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt, compile: true, sync: :step) { 0 }

    losses = trainer.fit([{ x: 1 }], epochs: 1)

    assert_equal 1, losses.length
    assert_equal({ optimizer: :opt, clip_grad_norm: nil }, model.train_step_kwargs)
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

  def test_fit_supports_zero_arity_dataset_factory
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    calls = 0
    dataset = lambda do
      calls += 1
      [{ x: calls }]
    end

    report = trainer.fit_report(dataset, epochs: 2)

    assert_equal 2, calls
    assert_equal 1, report.fetch("epochs")[0].fetch("batches")
    assert_equal 1, report.fetch("epochs")[1].fetch("batches")
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

  def test_fit_supports_dsl_data_pipeline_dataset
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    dataset = MLX::DSL::Data
      .from([1, 2, 3, 4])
      .filter { |x| x.even? }
      .map { |x| { x: x * 10 } }

    trainer.fit(dataset, epochs: 1)

    assert_equal [[], { x: 20 }], model.step.calls[0]
    assert_equal [[], { x: 40 }], model.step.calls[1]
  end

  def test_fit_supports_builtin_xy_collate
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.fit([[10, 20], [30, 40]], epochs: 1, collate: :xy)

    assert_equal [[], { x: 10, y: 20 }], model.step.calls[0]
    assert_equal [[], { x: 30, y: 40 }], model.step.calls[1]
  end

  def test_fit_supports_hash_collate_mapping
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.fit([[10, 20, 30]], epochs: 1, collate: { y: 1, x: 0, z: 2 })

    assert_equal [[], { y: 20, x: 10, z: 30 }], model.step.calls[0]
  end

  def test_batch_schema_supports_auto_collate_for_training
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }
    trainer.batch_schema(x: 0, y: 1)

    trainer.fit([[10, 20]], epochs: 1, collate: :auto)

    assert_equal [[], { x: 10, y: 20 }], model.step.calls[0]
  end

  def test_batch_schema_supports_split_specific_auto_collate
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) do |x:, y:|
      x.to_f + y.to_f
    end
    trainer.batch_schema(
      train: { x: 0, y: 1 },
      validation: { x: :a, y: :b }
    )

    report = trainer.fit_report(
      [[2.0, 3.0]],
      epochs: 1,
      collate: :auto,
      validation_data: [{ a: 5.0, b: 7.0 }],
      validation_collate: :auto
    )

    assert_equal [[], { x: 2.0, y: 3.0 }], model.step.calls[0]
    assert_equal 12.0, report.fetch("epochs")[0].fetch("val_loss")
  end

  def test_explicit_collate_takes_precedence_over_batch_schema_auto_behavior
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }
    trainer.batch_schema(x: 0, y: 1)

    trainer.fit([[10, 20]], epochs: 1, collate: { x: 1, y: 0 })

    assert_equal [[], { x: 20, y: 10 }], model.step.calls[0]
  end

  def test_use_dataflow_applies_named_profile_to_train_and_validation_loops
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) do |x:, y:|
      x.to_f + y.to_f
    end
    trainer.register_dataflow(
      :xy_profile,
      train: { collate: { x: 0, y: 1 }, limit: 1, reduce: :sum },
      validation: { collate: { x: :a, y: :b }, limit: 1, reduce: :mean }
    )

    report = trainer.fit_report(
      [[2.0, 3.0], [8.0, 9.0]],
      epochs: 1,
      validation_data: [{ a: 5.0, b: 7.0 }, { a: 11.0, b: 13.0 }],
      **trainer.use_dataflow(:xy_profile)
    )

    assert_equal [[], { x: 2.0, y: 3.0 }], model.step.calls[0]
    assert_equal 1, report.fetch("epochs")[0].fetch("batches")
    assert_equal 12.0, report.fetch("epochs")[0].fetch("val_loss")
  end

  def test_dataflow_profile_supports_extends_and_override_precedence
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0), FakeLoss.new(3.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }
    trainer.register_dataflow(
      :base_flow,
      train: { collate: { x: 0 }, limit: 1 },
      validation: { limit: 1, reduce: :mean }
    )
    trainer.register_dataflow(
      :extended_flow,
      train: { limit: 2 },
      extends: :base_flow
    )

    report = trainer.fit_report(
      [[10], [20], [30]],
      epochs: 1,
      **trainer.use_dataflow(:extended_flow),
      limit: 1
    )

    assert_equal [[], { x: 10 }], model.step.calls[0]
    assert_equal 1, report.fetch("epochs")[0].fetch("batches")
  end

  def test_fit_supports_hash_collate_mapping_with_proc_selectors
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.fit(
      [{ payload: [10, 20] }],
      epochs: 1,
      collate: {
        x: ->(batch) { batch.fetch(:payload).fetch(0) },
        y: ->(batch) { batch.fetch(:payload).fetch(1) }
      }
    )

    assert_equal [[], { x: 10, y: 20 }], model.step.calls[0]
  end

  def test_fit_supports_hash_collate_mapping_with_nested_path_selectors
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.fit(
      [{ "input" => { "x" => 10 }, target: [0, 20] }],
      epochs: 1,
      collate: {
        x: ["input", "x"],
        y: [:target, 1]
      }
    )

    assert_equal [[], { x: 10, y: 20 }], model.step.calls[0]
  end

  def test_fit_supports_collate_proc_with_context_arguments
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    seen = []
    trainer.fit(
      [[10, 20], [30, 40]],
      epochs: 1,
      collate: lambda do |batch, epoch:, batch_index:, kind:, trainer:|
        seen << [batch, epoch, batch_index, kind, trainer.class.name]
        { x: "#{batch[0]}:#{epoch}:#{batch_index}:#{kind}", y: batch[1] }
      end
    )

    assert_equal [[], { x: "10:0:0:train", y: 20 }], model.step.calls[0]
    assert_equal [[], { x: "30:0:1:train", y: 40 }], model.step.calls[1]
    assert_equal [[[10, 20], 0, 0, :train, "MLX::DSL::Trainer"], [[30, 40], 0, 1, :train, "MLX::DSL::Trainer"]], seen
  end

  def test_fit_supports_hash_collate_mapping_with_context_aware_proc_selectors
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.fit(
      [{ payload: [10, 20] }],
      epochs: 1,
      collate: {
        x: ->(batch, epoch:, batch_index:, kind:) { "#{batch.fetch(:payload).fetch(0)}-#{epoch}-#{batch_index}-#{kind}" },
        y: ->(batch, trainer:) { [batch.fetch(:payload).fetch(1), trainer.class.name] }
      }
    )

    assert_equal [[], { x: "10-0-0-train", y: [20, "MLX::DSL::Trainer"] }], model.step.calls[0]
  end

  def test_fit_report_supports_validation_collate_proc_with_context_arguments
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) do |x:, y:|
      x.to_f + y.to_f
    end

    seen = []
    report = trainer.fit_report(
      [{ x: 0, y: 0 }],
      epochs: 1,
      validation_data: [[2, 3]],
      validation_collate: lambda do |batch, epoch:, batch_index:, kind:|
        seen << [epoch, batch_index, kind]
        { x: batch[0], y: batch[1] }
      end
    )

    assert_equal [[0, 0, :validation]], seen
    assert_equal 5.0, report.fetch("epochs")[0].fetch("val_loss")
  end

  def test_fit_report_supports_validation_collate
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { |x:, y:| x + y }

    report = trainer.fit_report(
      [{ x: 0, y: 0 }],
      epochs: 1,
      validation_data: [[1.0, 3.0], [2.0, 4.0]],
      validation_collate: :xy
    )

    assert_equal 5.0, report.fetch("epochs")[0].fetch("val_loss")
  end

  def test_register_collate_supports_named_train_schema
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.register_collate(:pair_xy, { x: 0, y: 1 })
    trainer.fit([[10, 20]], epochs: 1, collate: :pair_xy)

    assert_equal [[], { x: 10, y: 20 }], model.step.calls[0]
  end

  def test_register_collate_supports_named_validation_schema_reuse
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) do |x:, y:|
      x.to_f + y.to_f
    end

    trainer.register_collate(:pair_xy, { x: 0, y: 1 })
    report = trainer.fit_report(
      [{ x: 0, y: 0 }],
      epochs: 1,
      validation_data: [[2.0, 3.0]],
      validation_collate: :pair_xy
    )

    assert_equal 5.0, report.fetch("epochs")[0].fetch("val_loss")
  end

  def test_register_collate_supports_hash_composition_with_extends
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.register_collate(:base_xy, { x: 0, y: 1 })
    trainer.register_collate(:with_meta, { meta: 2 }, extends: :base_xy)
    trainer.fit([[10, 20, 30]], epochs: 1, collate: :with_meta)

    assert_equal [[], { x: 10, y: 20, meta: 30 }], model.step.calls[0]
  end

  def test_register_collate_supports_multiple_extends_composition
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.register_collate(:base_a, { x: 0, a: 3 })
    trainer.register_collate(:base_b, { y: 1, x: 2 })
    trainer.register_collate(:combined, { z: 3 }, extends: [:base_a, :base_b])
    trainer.fit([[10, 20, 30, 40]], epochs: 1, collate: :combined)

    assert_equal [[], { x: 30, a: 40, y: 20, z: 40 }], model.step.calls[0]
  end

  def test_register_collate_rejects_unknown_base_for_multiple_extends
    trainer = MLX::DSL::Trainer.new(model: FakeModel.new([FakeLoss.new(1.0)]), optimizer: :opt) { 0 }
    trainer.register_collate(:base_a, { x: 0 })

    error = assert_raises(ArgumentError) do
      trainer.register_collate(:broken, { y: 1 }, extends: [:base_a, :missing])
    end
    assert_match(/unknown base collate/i, error.message)
  end

  def test_fit_report_supports_validation_limit
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { |x:| x.to_f }

    report = trainer.fit_report(
      [{ x: 0 }],
      epochs: 1,
      validation_data: [{ x: 1.0 }, { x: 3.0 }],
      validation_limit: 1,
      validation_reduce: :mean
    )

    assert_equal 1, report.fetch("epochs")[0].fetch("validation_batches")
    assert_equal 1.0, report.fetch("epochs")[0].fetch("val_loss")
  end

  def test_fit_supports_callable_train_limit_per_epoch
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(2.0), FakeLoss.new(3.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    seen = []
    report = trainer.fit_report(
      [{ x: 1 }, { x: 2 }, { x: 3 }],
      epochs: 2,
      limit: lambda do |epoch:, kind:, trainer:|
        seen << [epoch, kind, trainer.class.name]
        epoch.zero? ? 1 : 2
      end
    )

    assert_equal [1, 2], report.fetch("epochs").map { |row| row.fetch("batches") }
    assert_equal [[0, :train, "MLX::DSL::Trainer"], [1, :train, "MLX::DSL::Trainer"]], seen
  end

  def test_fit_report_supports_callable_validation_limit_per_epoch
    model = FakeModel.new([FakeLoss.new(1.0), FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { |x:| x.to_f }

    seen = []
    report = trainer.fit_report(
      [{ x: 0 }],
      epochs: 2,
      validation_data: [{ x: 1.0 }, { x: 2.0 }, { x: 3.0 }],
      validation_limit: lambda do |epoch:, kind:, trainer:|
        seen << [epoch, kind, trainer.class.name]
        epoch.zero? ? 1 : 2
      end,
      validation_reduce: :mean
    )

    assert_equal [1, 2], report.fetch("epochs").map { |row| row.fetch("validation_batches") }
    assert_equal 1.0, report.fetch("epochs")[0].fetch("val_loss")
    assert_equal 1.5, report.fetch("epochs")[1].fetch("val_loss")
    assert_equal [[0, :validation, "MLX::DSL::Trainer"], [1, :validation, "MLX::DSL::Trainer"]], seen
  end

  def test_fit_normalizes_string_hash_keys_for_keyword_train_step
    model = StrictKeywordModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.fit([{ "x" => 10, "y" => 20 }], epochs: 1)

    assert_equal [[10, 20]], model.step.calls
  end

  def test_fit_supports_bind_auto_for_common_signature_aliases
    model = StrictKeywordModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.fit([{ input: 10, target: 20 }], epochs: 1, bind: :auto)

    assert_equal [[10, 20]], model.step.calls
  end

  def test_fit_supports_bind_keypath_mapping_for_nested_batches
    model = StrictKeywordModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.fit(
      [{ payload: { features: 11, label: 22 } }],
      epochs: 1,
      bind: {
        x: [:payload, :features],
        y: [:payload, :label]
      }
    )

    assert_equal [[11, 22]], model.step.calls
  end

  def test_fit_explicit_collate_takes_precedence_over_bind
    model = StrictKeywordModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    trainer.fit(
      [[10, 20]],
      epochs: 1,
      collate: { x: 1, y: 0 },
      bind: { x: 0, y: 1 }
    )

    assert_equal [[20, 10]], model.step.calls
  end

  def test_fit_report_supports_validation_bind_mapping
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) do |x:, y:|
      x.to_f + y.to_f
    end

    report = trainer.fit_report(
      [{ x: 0, y: 0 }],
      epochs: 1,
      validation_data: [{ payload: { features: 1.5, label: 2.5 } }],
      validation_bind: { x: [:payload, :features], y: [:payload, :label] }
    )

    assert_equal 4.0, report.fetch("epochs")[0].fetch("val_loss")
  end

  def test_fit_report_normalizes_string_hash_keys_for_keyword_validation_loss
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { |x:, y:| x.to_f + y.to_f }

    report = trainer.fit_report(
      [{ x: 0, y: 0 }],
      epochs: 1,
      validation_data: [{ "x" => 1.0, "y" => 3.0 }]
    )

    assert_equal 4.0, report.fetch("epochs")[0].fetch("val_loss")
  end

  def test_fit_rejects_duplicate_keys_after_keyword_normalization
    model = StrictKeywordModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    error = assert_raises(ArgumentError) do
      trainer.fit([{ "x" => 1, x: 2, y: 3 }], epochs: 1)
    end
    assert_match(/duplicate keyword/, error.message)
  end

  def test_fit_rejects_unknown_collate_spec
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    error = assert_raises(ArgumentError) do
      trainer.fit([{ x: 1 }], epochs: 1, collate: :unknown_collate)
    end
    assert_match(/collate/, error.message)
  end

  def test_fit_includes_batch_context_when_train_step_raises
    model = RaisingTrainStepModel.new(ArgumentError.new("bad payload"))
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) { 0 }

    error = assert_raises(ArgumentError) do
      trainer.fit([{ x: 1 }], epochs: 1)
    end

    assert_match(/train batch failed at epoch 0, batch 0/i, error.message)
    assert_match(/bad payload/, error.message)
  end

  def test_fit_report_includes_batch_context_when_validation_raises
    model = FakeModel.new([FakeLoss.new(1.0)])
    trainer = MLX::DSL::Trainer.new(model: model, optimizer: :opt) do |x:|
      raise RuntimeError, "validation exploded with #{x.inspect}"
    end

    error = assert_raises(RuntimeError) do
      trainer.fit_report(
        [{ x: 0 }],
        epochs: 1,
        validation_data: [{ x: 42 }]
      )
    end

    assert_match(/validation batch failed at epoch 0, batch 0/i, error.message)
    assert_match(/validation exploded/, error.message)
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

  private

  def with_stubbed_core_eval
    core_singleton = class << MLX::Core
      self
    end

    remove_singleton_method = lambda do |name|
      next unless core_singleton.private_instance_methods(false).include?(name) ||
        core_singleton.protected_instance_methods(false).include?(name) ||
        core_singleton.instance_methods(false).include?(name)

      core_singleton.send(:remove_method, name)
    end
    eval_visibility =
      if core_singleton.private_instance_methods(false).include?(:eval)
        :private
      elsif core_singleton.protected_instance_methods(false).include?(:eval)
        :protected
      elsif core_singleton.instance_methods(false).include?(:eval)
        :public
      end

    remove_singleton_method.call(:__dsl_original_eval)
    if eval_visibility
      core_singleton.send(:alias_method, :__dsl_original_eval, :eval)
      remove_singleton_method.call(:eval)
    end

    calls = []
    core_singleton.define_method(:eval) do |*args|
      calls << args
      nil
    end

    yield calls
  ensure
    remove_singleton_method.call(:eval) if defined?(remove_singleton_method)

    if defined?(eval_visibility) && eval_visibility
      core_singleton.send(:alias_method, :eval, :__dsl_original_eval)
      core_singleton.send(:remove_method, :__dsl_original_eval)
      core_singleton.send(:private, :eval) if eval_visibility == :private
      core_singleton.send(:protected, :eval) if eval_visibility == :protected
    else
      remove_singleton_method.call(:eval) if defined?(remove_singleton_method)
    end
  end
end

$LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
