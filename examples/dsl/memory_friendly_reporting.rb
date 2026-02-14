# frozen_string_literal: true

require "mlx"

class MemoryAffine < MLX::DSL::Model
  option :in_dim
  option :out_dim

  param :weight, shape: -> { [out_dim, in_dim] }, init: ->(shape, _dtype) { MLX::Core.ones(shape, MLX::Core.float32) }
  buffer :offset, shape: -> { [out_dim] }, init: ->(shape) { MLX::Core.zeros(shape, MLX::Core.float32) }

  def call(x)
    MLX::Core.add(MLX::Core.matmul(x, weight.T), offset)
  end
end

model = MemoryAffine.new(in_dim: 1, out_dim: 1)
optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.02)

trainer = model.trainer(optimizer: optimizer) do |x:, y:|
  pred = model.call(x)
  MLX::NN.mse_loss(pred, y, reduction: "mean")
end

raw_batches = Array.new(20) { |i| [i.to_f, 0.0] }

report = trainer.fit_report(
  raw_batches,
  epochs: 2,
  keep_losses: false,
  train_transform: lambda do |batch, epoch:, batch_index:|
    _ = [epoch, batch_index]
    {
      x: MLX::Core.array([[batch[0]]], MLX::Core.float32),
      y: MLX::Core.array([[batch[1]]], MLX::Core.float32)
    }
  end
)

puts "losses_kept=#{report.fetch('losses_kept')} losses_count=#{report.fetch('losses').length}"
