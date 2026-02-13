# frozen_string_literal: true

require_relative "test_helper"
require "tempfile"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class DslTest < Minitest::Test
  def setup
    begin
      TestSupport.build_native_extension!
    rescue StandardError => e
      skip "Native extension build unavailable in this environment: #{e.message.lines.first&.strip || e.message}"
    end
    skip "MLX native extension unavailable" unless MLX.native_available?
  end

  class DslClassifier < MLX::DSL::Model
    option :in_dim
    option :hidden_dim, default: 8
    option :out_dim
    option :dropout_p, default: 0.0

    layer :net do
      sequential do
        linear in_dim, hidden_dim
        relu
        dropout dropout_p if dropout_p > 0
        linear hidden_dim, out_dim
      end
    end

    def call(x)
      net.call(x)
    end
  end

  class DslAffine < MLX::DSL::Model
    option :in_dim
    option :out_dim

    param :weight, shape: -> { [out_dim, in_dim] }, init: ->(shape, _dtype) { MLX::Core.ones(shape, MLX::Core.float32) }
    buffer :offset, shape: -> { [out_dim] }, init: ->(shape) { MLX::Core.zeros(shape, MLX::Core.float32) }

    def call(x)
      MLX::Core.add(MLX::Core.matmul(x, weight.T), offset)
    end
  end

  class DslMixinBlock < MLX::NN::Module
    include MLX::DSL::ModelMixin

    option :dims, default: 4
    layer(:proj) { linear dims, dims, bias: false }
    layer(:norm) { layer_norm dims }

    def call(x)
      norm.call(proj.call(x))
    end
  end

  class DslFreezeModel < MLX::DSL::Model
    option :in_dim
    option :hidden_dim
    option :out_dim

    layer(:encoder) { linear in_dim, hidden_dim }
    layer(:head) { linear hidden_dim, out_dim }

    def call(x)
      head.call(encoder.call(x))
    end
  end

  class DslRequiredOptionModel < MLX::DSL::Model
    option :dims, required: true
    layer(:proj) { linear dims, dims }

    def call(x)
      proj.call(x)
    end
  end

  class DslGraphModel < MLX::DSL::Model
    option :dims, default: 4

    layer :merge do
      concat(axis: -1) do
        identity
        residual do
          identity
        end
      end
    end

    def call(x)
      merge.call(x)
    end
  end

  class DslNetworkAliasModel < MLX::DSL::Model
    option :dims, default: 4

    network :stack do
      sequential do
        linear dims, dims
        relu
      end
    end

    def call(x)
      stack.call(x)
    end
  end

  def test_model_declarations_build_layers_and_options
    model = DslClassifier.new(in_dim: 4, out_dim: 2, hidden_dim: 6)
    assert_instance_of MLX::NN::Sequential, model.net
    assert_equal 4, model.in_dim
    assert_equal 6, model.hidden_dim
    assert_equal 2, model.out_dim

    x = MLX::Core.zeros([3, 4], MLX::Core.float32)
    y = model.call(x)
    assert_equal [3, 2], y.shape
  end

  def test_param_and_buffer_declarations_are_registered
    model = DslAffine.new(in_dim: 3, out_dim: 2)
    params = model.parameters
    trainable = model.trainable_parameters

    assert params.key?("weight")
    assert params.key?("offset")
    assert trainable.key?("weight")
    refute trainable.key?("offset")

    x = MLX::Core.ones([5, 3], MLX::Core.float32)
    y = model.call(x)
    assert_equal [5, 2], y.shape
  end

  def test_model_mixin_supports_layer_declarations
    model = DslMixinBlock.new(dims: 4)
    x = MLX::Core.ones([2, 4], MLX::Core.float32)
    y = model.call(x)
    assert_equal [2, 4], y.shape
  end

  def test_train_step_supports_hooks_and_updates_params
    model = DslAffine.new(in_dim: 1, out_dim: 1)
    optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.1)
    seen_steps = []

    step = model.train_step(optimizer: optimizer, clip_grad_norm: 1.0) do |x:, y:|
      pred = model.call(x)
      MLX::NN.mse_loss(pred, y, reduction: "mean")
    end
    step.on(:after_step) { |ctx| seen_steps << ctx[:step] }

    input = MLX::Core.array([[1.0], [2.0], [3.0]], MLX::Core.float32)
    target = MLX::Core.array([[0.0], [0.0], [0.0]], MLX::Core.float32)

    before = model.weight.to_a
    loss = step.call(x: input, y: target)
    after = model.weight.to_a

    assert_instance_of MLX::Core::Array, loss
    assert_equal [0], seen_steps
    refute_equal before, after
  end

  def test_freeze_and_unfreeze_by_path
    model = DslFreezeModel.new(in_dim: 3, hidden_dim: 5, out_dim: 2)

    trainable_before = MLX::Utils.tree_flatten(model.trainable_parameters, destination: {}).keys.sort
    assert_includes trainable_before, "encoder.weight"
    assert_includes trainable_before, "head.weight"

    model.freeze_paths!(/^encoder\./)
    trainable_mid = MLX::Utils.tree_flatten(model.trainable_parameters, destination: {}).keys.sort
    refute_includes trainable_mid, "encoder.weight"
    refute_includes trainable_mid, "encoder.bias"
    assert_includes trainable_mid, "head.weight"

    model.unfreeze_paths!(/^encoder\./)
    trainable_after = MLX::Utils.tree_flatten(model.trainable_parameters, destination: {}).keys.sort
    assert_includes trainable_after, "encoder.weight"
    assert_includes trainable_after, "encoder.bias"
  end

  def test_optimizer_groups_builder
    model = DslFreezeModel.new(in_dim: 3, hidden_dim: 5, out_dim: 2)
    optimizer = model.optimizer_groups do
      group(/^encoder\./) { MLX::Optimizers::AdamW.new(learning_rate: 1e-3) }
      group(nil) { MLX::Optimizers::SGD.new(learning_rate: 1e-2) }
    end

    assert_instance_of MLX::Optimizers::MultiOptimizer, optimizer
  end

  def test_required_option_validation
    assert_raises(ArgumentError) do
      DslRequiredOptionModel.new
    end
  end

  def test_graph_builder_helpers_residual_and_concat
    model = DslGraphModel.new(dims: 3)
    x = MLX::Core.array([[1.0, 2.0, 3.0]], MLX::Core.float32)
    y = model.call(x)
    assert_equal [1, 6], y.shape
    assert_nested_close [[1.0, 2.0, 3.0, 2.0, 4.0, 6.0]], y.to_a
  end

  def test_network_alias_behaves_like_layer
    model = DslNetworkAliasModel.new(dims: 5)
    x = MLX::Core.ones([2, 5], MLX::Core.float32)
    y = model.call(x)
    assert_equal [2, 5], y.shape
  end

  def test_checkpoint_roundtrip_model_and_optimizer
    model = DslAffine.new(in_dim: 1, out_dim: 1)
    optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.1)
    step = model.train_step(optimizer: optimizer) do |x:, y:|
      pred = model.call(x)
      MLX::NN.mse_loss(pred, y, reduction: "mean")
    end

    input = MLX::Core.array([[1.0], [2.0]], MLX::Core.float32)
    target = MLX::Core.array([[0.0], [0.0]], MLX::Core.float32)
    _loss = step.call(x: input, y: target)

    Tempfile.create(["mlx-dsl-checkpoint", ".bin"]) do |f|
      model.save_checkpoint(f.path, optimizer: optimizer, metadata: { "tag" => "dsl-test" })

      restored_model = DslAffine.new(in_dim: 1, out_dim: 1)
      restored_optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.1)
      payload = restored_model.load_checkpoint(f.path, optimizer: restored_optimizer, strict: true)

      assert_equal "mlx_dsl_checkpoint_v1", payload.fetch("format")
      assert_equal "dsl-test", payload.fetch("metadata").fetch("tag")
      assert_nested_close model.weight.to_a, restored_model.weight.to_a
      assert_equal optimizer.step, restored_optimizer.step
    end
  end

  def test_trainer_runs_over_dataset
    model = DslAffine.new(in_dim: 1, out_dim: 1)
    optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.05)
    events = []

    trainer = model.trainer(optimizer: optimizer, clip_grad_norm: 1.0) do |x:, y:|
      pred = model.call(x)
      MLX::NN.mse_loss(pred, y, reduction: "mean")
    end
    trainer.on(:after_batch) { |ctx| events << [ctx[:epoch], ctx[:batch_index]] }

    dataset = [
      {
        x: MLX::Core.array([[1.0], [2.0]], MLX::Core.float32),
        y: MLX::Core.array([[0.0], [0.0]], MLX::Core.float32)
      },
      {
        x: MLX::Core.array([[3.0], [4.0]], MLX::Core.float32),
        y: MLX::Core.array([[0.0], [0.0]], MLX::Core.float32)
      }
    ]

    losses = trainer.fit(dataset, epochs: 2)
    assert_equal 4, losses.length
    assert_equal [[0, 0], [0, 1], [1, 0], [1, 1]], events
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

    value.flat_map { |v| flatten(v) }
  end

  def shape_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |v| shape_signature(v) })]
  end
end

$LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
