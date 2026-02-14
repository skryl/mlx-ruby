# frozen_string_literal: true

require "tempfile"
require "mlx"

class CompileAffine < MLX::DSL::Model
  option :in_dim
  option :out_dim

  param :weight, shape: -> { [out_dim, in_dim] }, init: ->(shape, _dtype) { MLX::Core.ones(shape, MLX::Core.float32) }
  buffer :offset, shape: -> { [out_dim] }, init: ->(shape) { MLX::Core.zeros(shape, MLX::Core.float32) }

  def call(x)
    MLX::Core.add(MLX::Core.matmul(x, weight.T), offset)
  end
end

model = CompileAffine.new(in_dim: 1, out_dim: 1)
optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.05)

compile_enabled = ENV["MLX_DSL_EXAMPLE_COMPILE"] == "1"
compile_options = if compile_enabled && MLX::Core.respond_to?(:compile)
  { shapeless: true }
else
  false
end

step = model.train_step(optimizer: optimizer, compile: compile_options, sync: :step) do |x:, y:|
  pred = model.call(x)
  MLX::NN.mse_loss(pred, y, reduction: "mean")
end

loss = step.call(
  x: MLX::Core.array([[1.0], [2.0]], MLX::Core.float32),
  y: MLX::Core.array([[0.0], [0.0]], MLX::Core.float32)
)

trainer = model.trainer(optimizer: optimizer, compile: compile_options, sync: :epoch) do |x:, y:|
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
  epochs: 2
)

Tempfile.create(["mlx-dsl-native-ckpt", ".npz"]) do |f|
  path = f.path
  f.close

  model.save_checkpoint(path, optimizer: optimizer, metadata: { "tag" => "compile-sync-example" })
  restored_model = CompileAffine.new(in_dim: 1, out_dim: 1)
  restored_optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.05)
  payload = restored_model.load_checkpoint(path, optimizer: restored_optimizer, strict: true)

  puts "checkpoint_format=#{payload.fetch('format')} weights_format=#{payload.fetch('weights_format')}"
  puts "restored_optimizer_step=#{restored_optimizer.step}"
end

puts "compiled=#{compile_options != false} step_loss=#{loss.item} best_metric=#{report.fetch('best_metric')}"
