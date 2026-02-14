# frozen_string_literal: true

require_relative "../test_helper"
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

  class DslFactoryClassArgsModel < MLX::DSL::Model
    option :dims, default: 4

    layer :proj, MLX::NN::Linear, -> { dims }, -> { dims }, bias: false

    def call(x)
      proj.call(x)
    end
  end

  class DslFactoryCallableArgsModel < MLX::DSL::Model
    option :dims, default: 4

    layer :proj,
          ->(in_dim, out_dim, bias: true) { MLX::NN::Linear.new(in_dim, out_dim, bias: bias) },
          -> { dims },
          -> { dims },
          bias: false

    def call(x)
      proj.call(x)
    end
  end

  class DslStackedModel < MLX::DSL::Model
    option :dims, default: 4

    layer :net do
      stack(3, MLX::NN::Linear, dims, dims)
    end

    def call(x)
      net.call(x)
    end
  end

  class DslOneShotDataset
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

  def test_layer_factory_supports_class_with_dynamic_constructor_args
    model = DslFactoryClassArgsModel.new(dims: 3)
    assert_instance_of MLX::NN::Linear, model.proj

    x = MLX::Core.ones([2, 3], MLX::Core.float32)
    y = model.call(x)
    assert_equal [2, 3], y.shape
  end

  def test_layer_factory_supports_callable_with_dynamic_args_and_kwargs
    model = DslFactoryCallableArgsModel.new(dims: 3)
    assert_instance_of MLX::NN::Linear, model.proj

    x = MLX::Core.ones([2, 3], MLX::Core.float32)
    y = model.call(x)
    assert_equal [2, 3], y.shape
  end

  def test_stack_builder_tracks_repeated_module_parameters
    model = DslStackedModel.new(dims: 3)
    assert_instance_of MLX::NN::Sequential, model.net

    paths = model.parameter_paths
    assert_equal 6, paths.length
    assert_includes paths, "net.layers.0.weight"
    assert_includes paths, "net.layers.1.weight"
    assert_includes paths, "net.layers.2.weight"

    x = MLX::Core.ones([2, 3], MLX::Core.float32)
    y = model.call(x)
    assert_equal [2, 3], y.shape
  end

  def test_layer_rejects_factory_and_block_together
    error = assert_raises(ArgumentError) do
      Class.new(MLX::DSL::Model) do
        layer :bad, MLX::NN::Identity do
          identity
        end
      end
    end

    assert_match(/factory and block/i, error.message)
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

  def test_model_introspection_helpers_report_counts_paths_and_summary
    model = DslAffine.new(in_dim: 3, out_dim: 2)

    assert_equal 8, model.parameter_count
    assert_equal 6, model.trainable_parameter_count
    assert_equal ["offset", "weight"], model.parameter_paths
    assert_equal ["weight"], model.parameter_paths(matcher: /^weight$/)

    summary = model.summary
    assert_equal "DslTest::DslAffine", summary.fetch("model_class")
    assert_equal 8, summary.fetch("total_parameters")
    assert_equal 6, summary.fetch("trainable_parameters")
    assert_equal 2, summary.fetch("frozen_parameters")
    assert_equal ["offset", "weight"], summary.fetch("parameter_paths")

    text = model.summary(as: :text)
    assert_match(/DslTest::DslAffine/, text)
    assert_match(/total_parameters=8/, text)
    assert_match(/trainable_parameters=6/, text)
    assert_match(/frozen_parameters=2/, text)
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

  def test_unknown_option_raises_clear_error
    klass = Class.new(MLX::DSL::Model) do
      option :dims
      layer(:proj) { linear dims, dims }

      def call(x)
        proj.call(x)
      end
    end

    error = assert_raises(ArgumentError) do
      klass.new(dims: 4, typo: true)
    end
    assert_match(/unknown option/i, error.message)
    assert_match(/typo/, error.message)
  end

  def test_layer_declaration_requires_module_result
    klass = Class.new(MLX::DSL::Model) do
      layer(:broken) { 123 }

      def call(x)
        x
      end
    end

    error = assert_raises(TypeError) do
      klass.new
    end
    assert_match(/MLX::NN::Module/, error.message)
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

  def test_builder_exposes_extended_nn_layer_helpers
    builder = MLX::DSL::Builder.new

    assert_instance_of MLX::NN::ConvTranspose1d, builder.conv_transpose1d(2, 2, 3)
    assert_instance_of MLX::NN::ConvTranspose2d, builder.conv_transpose2d(2, 2, 3)
    assert_instance_of MLX::NN::ConvTranspose3d, builder.conv_transpose3d(2, 2, 3)

    assert_instance_of MLX::NN::MaxPool1d, builder.max_pool1d(2)
    assert_instance_of MLX::NN::AvgPool1d, builder.avg_pool1d(2)
    assert_instance_of MLX::NN::MaxPool3d, builder.max_pool3d(2)
    assert_instance_of MLX::NN::AvgPool3d, builder.avg_pool3d(2)

    assert_instance_of MLX::NN::LeakyReLU, builder.leaky_relu(0.2)
    assert_instance_of MLX::NN::RNN, builder.rnn(4, 4)
    assert_instance_of MLX::NN::GRU, builder.gru(4, 4)
    assert_instance_of MLX::NN::LSTM, builder.lstm(4, 4)

    assert_instance_of MLX::NN::MultiHeadAttention, builder.multi_head_attention(4, 2)
    assert_instance_of MLX::NN::TransformerEncoderLayer, builder.transformer_encoder_layer(4, 2)
    assert_instance_of MLX::NN::TransformerEncoder, builder.transformer_encoder(1, 4, 2)
    assert_instance_of MLX::NN::TransformerDecoderLayer, builder.transformer_decoder_layer(4, 2)
    assert_instance_of MLX::NN::TransformerDecoder, builder.transformer_decoder(1, 4, 2)
    assert_instance_of MLX::NN::Transformer,
                       builder.transformer(dims: 4, num_heads: 2, num_encoder_layers: 1, num_decoder_layers: 1)

    assert_instance_of MLX::NN::RoPE, builder.rope(4)
    assert_instance_of MLX::NN::SinusoidalPositionalEncoding, builder.sinusoidal_positional_encoding(4)
    assert_instance_of MLX::NN::ALiBi, builder.alibi
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

  def test_checkpoint_roundtrip_native_npz_model_and_optimizer
    model = DslAffine.new(in_dim: 1, out_dim: 1)
    optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.1)
    step = model.train_step(optimizer: optimizer) do |x:, y:|
      pred = model.call(x)
      MLX::NN.mse_loss(pred, y, reduction: "mean")
    end

    input = MLX::Core.array([[1.0], [2.0]], MLX::Core.float32)
    target = MLX::Core.array([[0.0], [0.0]], MLX::Core.float32)
    _loss = step.call(x: input, y: target)

    Tempfile.create(["mlx-dsl-checkpoint-native", ".npz"]) do |f|
      path = f.path
      f.close

      model.save_checkpoint(path, optimizer: optimizer, metadata: { "tag" => "dsl-test-native" })
      assert File.exist?(path)
      assert File.exist?("#{path}.mlxmeta.json")

      restored_model = DslAffine.new(in_dim: 1, out_dim: 1)
      restored_optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.1)
      payload = restored_model.load_checkpoint(path, optimizer: restored_optimizer, strict: true)

      assert_equal "mlx_dsl_checkpoint_v2_native", payload.fetch("format")
      assert_equal "npz", payload.fetch("weights_format")
      assert_equal "dsl-test-native", payload.fetch("metadata").fetch("tag")
      assert_nested_close model.weight.to_a, restored_model.weight.to_a
      assert_equal optimizer.step, restored_optimizer.step
    end
  end

  def test_load_checkpoint_autodetects_native_extension_for_extensionless_base_path
    model = DslAffine.new(in_dim: 1, out_dim: 1)
    optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.1)
    step = model.train_step(optimizer: optimizer) do |x:, y:|
      pred = model.call(x)
      MLX::NN.mse_loss(pred, y, reduction: "mean")
    end

    input = MLX::Core.array([[1.0], [2.0]], MLX::Core.float32)
    target = MLX::Core.array([[0.0], [0.0]], MLX::Core.float32)
    _loss = step.call(x: input, y: target)

    Dir.mktmpdir("mlx-dsl-checkpoint-autodetect") do |dir|
      base_path = File.join(dir, "checkpoint")
      weights_path = model.save_checkpoint(base_path, optimizer: optimizer, format: :npz, metadata: { "tag" => "auto-detect" })
      assert_equal "#{base_path}.npz", weights_path
      assert File.exist?(weights_path)

      restored_model = DslAffine.new(in_dim: 1, out_dim: 1)
      restored_optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.1)
      payload = restored_model.load_checkpoint(base_path, optimizer: restored_optimizer, strict: true)

      assert_equal "mlx_dsl_checkpoint_v2_native", payload.fetch("format")
      assert_equal "npz", payload.fetch("weights_format")
      assert_equal "auto-detect", payload.fetch("metadata").fetch("tag")
      assert_nested_close model.weight.to_a, restored_model.weight.to_a
      assert_equal optimizer.step, restored_optimizer.step
    end
  end

  def test_save_checkpoint_creates_parent_directories_for_marshal
    model = DslAffine.new(in_dim: 1, out_dim: 1)

    Dir.mktmpdir("mlx-dsl-checkpoint-dir") do |dir|
      path = File.join(dir, "nested", "marshal", "checkpoint.bin")
      model.save_checkpoint(path, metadata: { "tag" => "mkdir-marshal" })
      assert File.exist?(path)
    end
  end

  def test_save_checkpoint_creates_parent_directories_for_native_format
    model = DslAffine.new(in_dim: 1, out_dim: 1)

    Dir.mktmpdir("mlx-dsl-checkpoint-dir") do |dir|
      path = File.join(dir, "nested", "native", "checkpoint.npz")
      model.save_checkpoint(path, metadata: { "tag" => "mkdir-native" })
      assert File.exist?(path)
      assert File.exist?("#{path}.mlxmeta.json")
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

  def test_trainer_fit_report_supports_train_transform_and_keep_losses_false
    model = DslAffine.new(in_dim: 1, out_dim: 1)
    optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.05)
    seen = []

    trainer = model.trainer(optimizer: optimizer) do |x:, y:|
      pred = model.call(x)
      MLX::NN.mse_loss(pred, y, reduction: "mean")
    end

    report = trainer.fit_report(
      [[1.0, 0.0], [2.0, 0.0]],
      epochs: 1,
      keep_losses: false,
      train_transform: lambda do |batch, epoch:, batch_index:, kind:, trainer:|
        seen << [epoch, batch_index, kind, trainer.class.name]
        {
          x: MLX::Core.array([[batch[0]]], MLX::Core.float32),
          y: MLX::Core.array([[batch[1]]], MLX::Core.float32)
        }
      end
    )

    assert_equal false, report.fetch("losses_kept")
    assert_equal [], report.fetch("losses")
    assert_equal 2, report.fetch("epochs")[0].fetch("batches")
    assert_equal [[0, 0, :train, "MLX::DSL::Trainer"], [0, 1, :train, "MLX::DSL::Trainer"]], seen
  end

  def test_trainer_fit_report_supports_validation_transform_and_val_monitor
    model = DslAffine.new(in_dim: 1, out_dim: 1)
    optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.01)

    trainer = model.trainer(optimizer: optimizer) do |x:, y:|
      pred = model.call(x)
      MLX::NN.mse_loss(pred, y, reduction: "mean")
    end

    report = trainer.fit_report(
      [
        {
          x: MLX::Core.array([[1.0]], MLX::Core.float32),
          y: MLX::Core.array([[0.0]], MLX::Core.float32)
        }
      ],
      epochs: 1,
      validation_data: [[1.0, 0.0], [3.0, 0.0]],
      validation_transform: lambda do |batch, epoch:|
        {
          x: MLX::Core.array([[batch[0] + epoch]], MLX::Core.float32),
          y: MLX::Core.array([[batch[1]]], MLX::Core.float32)
        }
      end,
      monitor: :val_loss,
      monitor_mode: :min
    )

    epoch_row = report.fetch("epochs")[0]
    assert_equal "val_loss", report.fetch("monitor_name")
    refute_nil epoch_row.fetch("val_loss")
    assert_in_delta epoch_row.fetch("val_loss"), epoch_row.fetch("monitor_value"), 1e-6
    assert_equal 2, epoch_row.fetch("validation_batches")
  end

  def test_trainer_strict_data_reuse_raises_for_exhausted_dataset
    model = DslAffine.new(in_dim: 1, out_dim: 1)
    optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.01)

    trainer = model.trainer(optimizer: optimizer) do |x:, y:|
      pred = model.call(x)
      MLX::NN.mse_loss(pred, y, reduction: "mean")
    end

    one_shot = DslOneShotDataset.new(
      [
        {
          x: MLX::Core.array([[1.0]], MLX::Core.float32),
          y: MLX::Core.array([[0.0]], MLX::Core.float32)
        }
      ]
    )

    error = assert_raises(ArgumentError) do
      trainer.fit_report(one_shot, epochs: 2, strict_data_reuse: true)
    end
    assert_match(/exhausted/i, error.message)
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
